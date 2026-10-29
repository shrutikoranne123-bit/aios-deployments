"""
retrieval.py — Multi-Backend RAG Retrieval Pipeline
=====================================================
Supports all vector store backends:
  1  → Qdrant (internal)
  2  → Qdrant (user-provided)
  3  → Pinecone
  4  → Redis
  5  → PGVector (PostgreSQL)
  6  → MongoDB Atlas

Search modes (all backends support dense; hybrid/sparse where available):
  dense      — dense vector search only
  sparse     — SPLADE sparse search only  (Qdrant, Pinecone native; others: skipped)
  hybrid     — client-side RRF over dense + sparse
  hybrid_rrf — alias of hybrid

The backend is selected from `vector_store_details.vector_store_id` in the
request payload, mirroring exactly how ingestion selects its backend.
"""


import argparse
import json
import logging
import os
import re
import time
import requests
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# Load environment variables from the project-root .env before reading any
# os.getenv() config below. Needed so boto3 (query embedding via AWS Bedrock)
# and OpenAI credentials are available when this module runs as a server.
try:
    from dotenv import load_dotenv

    # Prefer the retrieval-local .env (<this_folder>/.env); fall back to the
    # project-root .env one level up, then to an upward search. This keeps the
    # retrieval service self-contained while still working if only the root
    # .env exists.
    load_dotenv()
except ImportError:
    pass  # dotenv optional; env may already be set by the runtime

logger = logging.getLogger(__name__)

# FastAPI is imported lazily inside create_app() so the file can still be
# used as a plain CLI script without fastapi/uvicorn installed.

# ========================= CONFIG =========================

EMBED_BASE      = "https://fileupload-alpha.q0.dev/q0/embedding"
SPARSE_EMBED_URL = "https://fileupload-alpha.q0.dev/q0/splade"
DEFAULT_DENSE_MODEL = "bge-large"

# AWS Bedrock config for query embedding
BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "cohere.embed-multilingual-v3")
BEDROCK_EMBED_ENABLED = os.getenv("BEDROCK_EMBED_ENABLED", "1").lower() in ("1", "true", "yes")

# AWS Bedrock config for answer generation (Anthropic Claude on Bedrock).
# Falls back to the vision model id that is already enabled in this account.
BEDROCK_CHAT_MODEL = os.getenv(
    "BEDROCK_CHAT_MODEL",
    os.getenv("VISION_BEDROCK_MODEL", "us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
)
BEDROCK_CHAT_MAX_TOKENS = int(os.getenv("BEDROCK_CHAT_MAX_TOKENS", "1024"))

# OpenAI config kept only as an optional fallback; Bedrock is the default.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL")

QDRANT_URLS = {
    "dev": "https://fileupload-alpha.q0.dev/qdrant-dev",
    "uat": "https://fileupload-uat.q0.dev/qdrant-uat",
}

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"
RRF_K = 60

# Llama-3 special tokens
BOS            = "<|begin_of_text|>"
SOH            = "<|start_header_id|>"
EOH            = "<|end_header_id|>"
EOT            = "<|eot_id|>"
ASSISTANT_ROLE = "assistant"


# ========================= DATA CLASS =========================

@dataclass
class RetrievalResult:
    score:         float
    text:          str
    metadata:      dict = field(default_factory=dict)
    point_id:      Optional[str] = None
    dense_values:  Optional[List[float]] = None
    sparse_values: Optional[Dict[str, List]] = None


class RetrievalExecutionError(RuntimeError):
    """Raised when retrieval fails due to backend or connectivity issues."""


# ========================= EMBEDDINGS =========================

def _is_int_like(value: Any) -> bool:
    try:
        int(value)
        return True
    except (TypeError, ValueError):
        return False


def _is_float_like(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _normalize_sparse_vector(indices: List[Any], values: List[Any]) -> Dict[str, List]:
    """Deduplicate and sort sparse indices/values."""
    if len(indices) != len(values):
        raise ValueError("Sparse vector indices and values must have the same length")
    aggregated: Dict[int, float] = {}
    for raw_index, raw_value in zip(indices, values):
        try:
            idx = int(raw_index)
            val = float(raw_value)
        except (TypeError, ValueError):
            continue
        if val == 0.0:
            continue
        aggregated[idx] = aggregated.get(idx, 0.0) + val
    sorted_indices = sorted(aggregated.keys())
    return {"indices": sorted_indices, "values": [aggregated[i] for i in sorted_indices]}


def _parse_sparse_candidate(candidate: Any) -> Optional[Dict[str, List]]:
    if not candidate:
        return None
    if isinstance(candidate, str):
        stripped = candidate.strip()
        if stripped.startswith(("{", "[")):
            try:
                return _parse_sparse_candidate(json.loads(stripped))
            except json.JSONDecodeError:
                return None
        return None
    if isinstance(candidate, dict):
        if "indices" in candidate and "values" in candidate:
            return _normalize_sparse_vector(candidate["indices"], candidate["values"])
        if candidate and all(
            _is_int_like(k) and _is_float_like(v) for k, v in candidate.items()
        ):
            return _normalize_sparse_vector(list(candidate.keys()), list(candidate.values()))
        return None
    if isinstance(candidate, list):
        if len(candidate) == 2 and all(isinstance(item, list) for item in candidate):
            return _normalize_sparse_vector(candidate[0], candidate[1])
        if len(candidate) == 1:
            return _parse_sparse_candidate(candidate[0])
        if candidate and all(
            not isinstance(item, (list, dict)) and _is_float_like(item)
            for item in candidate
        ):
            return _normalize_sparse_vector(list(range(len(candidate))), candidate)
    return None


def _extract_sparse_embedding_response(data: Dict[str, Any]) -> Optional[Dict[str, List]]:
    if isinstance(data, dict):
        for key in ("sparse", "sparse_embedding", "embedding"):
            sv = _parse_sparse_candidate(data.get(key))
            if sv is not None:
                return sv
    outputs = data.get("outputs", [])
    if not isinstance(outputs, list):
        return None
    index_output = value_output = None
    for output in outputs:
        if not isinstance(output, dict):
            continue
        name = str(output.get("name", "")).upper()
        if "INDICES" in name and index_output is None:
            index_output = output.get("data", [])
        elif "VALUES" in name and value_output is None:
            value_output = output.get("data", [])
        sv = _parse_sparse_candidate(output.get("data"))
        if sv is not None:
            return sv
    if index_output is not None and value_output is not None:
        return _normalize_sparse_vector(index_output, value_output)
    return None


def _maybe_parse_json_metadata_value(value: Any) -> Any:
    """Decode JSON-looking metadata strings stored by Pinecone."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped.startswith(("{", "[")):
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def _normalize_pinecone_metadata(metadata: Any) -> Dict[str, Any]:
    """Restore structured metadata fields after Pinecone string sanitization."""
    if not isinstance(metadata, dict):
        return {}
    return {
        key: _maybe_parse_json_metadata_value(value)
        for key, value in metadata.items()
    }


def _normalize_pinecone_sparse_values(sparse_values: Any) -> Optional[Dict[str, List]]:
    """Normalize sparse values returned by Pinecone fetch/query APIs."""
    parsed = _parse_sparse_candidate(sparse_values)
    if parsed is not None:
        return parsed

    if sparse_values is None:
        return None

    indices = getattr(sparse_values, "indices", None)
    values = getattr(sparse_values, "values", None)
    if indices is None or values is None:
        return None

    return _normalize_sparse_vector(indices, values)


def _pinecone_description_field(description: Any, field: str) -> Any:
    """Read an index description field from dict- or object-style SDK responses."""
    if isinstance(description, dict):
        return description.get(field)
    return getattr(description, field, None)


def _pinecone_sparse_query_config(pc: Any, index_name: str) -> Tuple[str, str]:
    """Return the Pinecone index vector type and metric."""
    description = pc.describe_index(index_name)
    vector_type = str(
        _pinecone_description_field(description, "vector_type") or ""
    ).strip().lower()
    metric = str(
        _pinecone_description_field(description, "metric") or ""
    ).strip().lower()
    return vector_type, metric


def _pinecone_supports_sparse_query(pc: Any, index_name: str) -> bool:
    """Return True when the Pinecone index supports sparse query values."""
    vector_type, metric = _pinecone_sparse_query_config(pc, index_name)
    return vector_type == "sparse" or metric == "dotproduct"


def _validate_pinecone_sparse_query_support(pc: Any, index_name: str) -> None:
    """
    Ensure the target Pinecone index can accept sparse query values.

    Pinecone only supports sparse query values on:
    - sparse indexes, or
    - dense indexes using the dotproduct metric.
    """
    vector_type, metric = _pinecone_sparse_query_config(pc, index_name)
    if _pinecone_supports_sparse_query(pc, index_name):
        return

    raise ValueError(
        f"Pinecone index '{index_name}' is configured with metric "
        f"'{metric or 'unknown'}'. Sparse/hybrid queries require either "
        "a sparse index or a dense index with metric 'dotproduct'. "
        "Recreate the index with dotproduct and re-ingest the data."
    )


def _pinecone_index_supports_sparse_query(vs_details: Dict[str, Any]) -> bool:
    """Check whether the configured Pinecone index supports sparse query values."""
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise ImportError("pip install pinecone-client") from exc

    api_key = vs_details.get("PINECONE_API_KEY") or vs_details.get("pinecone_api_key")
    index_name = vs_details.get("PINECONE_INDEX_NAME") or vs_details.get("pinecone_index_name")

    pc = Pinecone(api_key=api_key)
    return _pinecone_supports_sparse_query(pc, index_name)


def _pinecone_fetch_by_ids(
    point_ids: List[str],
    vs_details: Dict[str, Any],
    collection: str,
    *,
    include_dense_values: bool = False,
    include_sparse_values: bool = False,
) -> List[RetrievalResult]:
    """Fetch stored Pinecone records by ID, optionally including vector values."""
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise ImportError("pip install pinecone-client") from exc

    api_key = vs_details.get("PINECONE_API_KEY") or vs_details.get("pinecone_api_key")
    index_name = vs_details.get("PINECONE_INDEX_NAME") or vs_details.get("pinecone_index_name")
    namespace = vs_details.get("PINECONE_NAMESPACE") or vs_details.get("pinecone_namespace") or collection

    unique_ids = [pid for pid in dict.fromkeys(point_ids) if pid]
    if not unique_ids:
        return []

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    fetched = index.fetch(ids=unique_ids, namespace=namespace)
    vectors_map = (
        fetched.get("vectors", {})
        if isinstance(fetched, dict)
        else getattr(fetched, "vectors", {})
    ) or {}

    results: List[RetrievalResult] = []
    for point_id in unique_ids:
        vector_data = vectors_map.get(point_id)
        if vector_data is None:
            continue

        if isinstance(vector_data, dict):
            metadata = _normalize_pinecone_metadata(vector_data.get("metadata") or {})
            dense_values = vector_data.get("values") if include_dense_values else None
            sparse_payload = (
                vector_data.get("sparse_values")
                or vector_data.get("sparseValues")
            )
        else:
            metadata = _normalize_pinecone_metadata(getattr(vector_data, "metadata", {}) or {})
            dense_values = getattr(vector_data, "values", None) if include_dense_values else None
            sparse_payload = (
                getattr(vector_data, "sparse_values", None)
                or getattr(vector_data, "sparseValues", None)
            )

        sparse_values = (
            _normalize_pinecone_sparse_values(sparse_payload)
            if include_sparse_values
            else None
        )

        results.append(RetrievalResult(
            score=1.0,
            text=metadata.get("text", ""),
            metadata=metadata,
            point_id=str(point_id),
            dense_values=dense_values,
            sparse_values=sparse_values,
        ))

    return results


def _enrich_pinecone_results_with_fetch(
    results: List[RetrievalResult],
    vs_details: Dict[str, Any],
    collection: str,
    *,
    include_dense_values: bool = False,
    include_sparse_values: bool = False,
) -> List[RetrievalResult]:
    """Attach stored Pinecone vector values to query results via fetch-by-id."""
    point_ids = [result.point_id for result in results if result.point_id]
    if not point_ids:
        return results

    fetched_results = _pinecone_fetch_by_ids(
        point_ids,
        vs_details,
        collection,
        include_dense_values=include_dense_values,
        include_sparse_values=include_sparse_values,
    )
    fetched_map = {result.point_id: result for result in fetched_results if result.point_id}

    for result in results:
        fetched = fetched_map.get(result.point_id)
        if fetched is None:
            continue
        if include_dense_values:
            result.dense_values = fetched.dense_values
        if include_sparse_values:
            result.sparse_values = fetched.sparse_values

    return results


def get_dense_embedding(text: str, model: str = DEFAULT_DENSE_MODEL) -> List[float]:
    """
    Generate dense embedding for a query.

    Uses AWS Bedrock Cohere Multilingual when BEDROCK_EMBED_ENABLED=1,
    otherwise falls back to the legacy cluster endpoint.
    """
    if BEDROCK_EMBED_ENABLED:
        return _bedrock_embed(text)

    # Legacy: cluster endpoint
    url  = f"{EMBED_BASE}/{model}"
    resp = requests.post(url, json={"model": model, "input": text}, timeout=45)
    if resp.status_code != 200:
        raise RuntimeError(f"Dense embedding failed ({resp.status_code}): {resp.text[:150]}")
    return resp.json()["data"][0]["embedding"]


def _bedrock_embed(text: str) -> List[float]:
    """Generate embedding using AWS Bedrock Cohere Multilingual v3."""
    import boto3

    client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    body = json.dumps({
        "texts": [text],
        "input_type": "search_query",
        "truncate": "END",
    })
    response = client.invoke_model(
        modelId=BEDROCK_EMBED_MODEL,
        body=body,
        contentType="application/json",
        accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embeddings"][0]


def get_sparse_embedding(text: str) -> Dict[str, List]:
    """
    Generate sparse embedding for a query.

    When Bedrock is enabled, returns empty sparse vector (no SPLADE call).
    Hybrid/sparse modes still "work" but sparse contribution is zero — effectively dense-only.
    """
    if BEDROCK_EMBED_ENABLED:
        return {"indices": [], "values": []}

    payload = {
        "inputs": [{"name": "TEXT", "shape": [1, 1], "datatype": "BYTES", "data": [text]}]
    }
    resp = requests.post(SPARSE_EMBED_URL, json=payload, timeout=45)
    if resp.status_code != 200:
        raise RuntimeError(f"Sparse embedding failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json()
    sv = _extract_sparse_embedding_response(data)
    if sv is None:
        raise RuntimeError(f"Unexpected sparse embedding response format: {data}")
    return sv


# ========================= RRF =========================

def reciprocal_rank_fusion(
    result_lists: List[List[RetrievalResult]],
    k: int = RRF_K,
) -> List[RetrievalResult]:
    rrf_scores: Dict[str, float]           = {}
    point_map:  Dict[str, RetrievalResult] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            rid = result.point_id or result.text[:64]
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (k + rank)
            point_map[rid]  = result
    ranked_ids = sorted(rrf_scores, key=lambda rid: rrf_scores[rid], reverse=True)
    fused: List[RetrievalResult] = []
    for rid in ranked_ids:
        result       = point_map[rid]
        result.score = rrf_scores[rid]
        fused.append(result)
    return fused


# ========================= QDRANT BACKEND =========================

def _qdrant_base(env_or_url: str) -> str:
    return QDRANT_URLS.get(env_or_url, env_or_url or QDRANT_URLS["uat"])


def _parse_qdrant_hits(hits: list) -> List[RetrievalResult]:
    return [
        RetrievalResult(
            score    = hit["score"],
            text     = (hit.get("payload") or {}).get("text", ""),
            metadata = hit.get("payload") or {},
            point_id = str(hit.get("id", "")),
        )
        for hit in hits
    ]


# ========================= FILTER UTILITIES =========================

def _qdrant_match_condition(key: str, value: Any) -> Dict[str, Any]:
    """Build a Qdrant equality condition, coercing numeric strings to integers.

    Qdrant's ``match.value`` is type-strict AND a filtered field can carry only
    one payload index type. Numeric metadata here (org_id, knowledge_base_id,
    etc.) is stored as integers during ingestion, but requests send them as
    strings (e.g. "53"). So a numeric-looking string is matched as an integer;
    everything else is matched as-is. This keeps a single, correct index type
    per field.
    """
    if not isinstance(value, bool) and isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            try:
                value = int(stripped)
            except ValueError:
                pass
    return {"key": key, "match": {"value": value}}


def _apply_is_active_filter(
    filters: Optional[Dict[str, Any]] = None,
    is_active: Optional[bool] = True,
    version: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Merge isActive, version, and user filters into a Qdrant filter dict.

    Converts simple key-value filters (e.g. {"org_id": "298"}) into proper
    Qdrant must conditions. Equality matches are type-tolerant so values stored
    as int or string both match (see _qdrant_match_condition).
    """
    # Build list of system conditions
    conditions_to_add = []

    if is_active is not None:
        conditions_to_add.append(_qdrant_match_condition("isActive", is_active))

    # Version filter: specific version or current only
    if version is not None:
        conditions_to_add.append({"key": "version_number", "match": {"value": version}})
    else:
        conditions_to_add.append({"key": "is_current_version", "match": {"value": True}})

    # Convert user filters to Qdrant must conditions
    if filters:
        if "must" in filters or "should" in filters or "must_not" in filters:
            # Already in Qdrant format — extract existing must conditions
            existing_must = filters.get("must", [])
            if isinstance(existing_must, list):
                conditions_to_add.extend(existing_must)
            elif existing_must:
                conditions_to_add.append(existing_must)
        else:
            # Simple key-value format — convert each to Qdrant condition
            for key, value in filters.items():
                if isinstance(value, dict):
                    # Operator format — convert to Qdrant range/match
                    if "$in" in value:
                        conditions_to_add.append({"key": key, "match": {"any": value["$in"]}})
                    elif "$eq" in value:
                        conditions_to_add.append(_qdrant_match_condition(key, value["$eq"]))
                    elif "$ne" in value:
                        # Qdrant doesn't have $ne directly; use must_not (handled separately)
                        pass
                    else:
                        # Range operators
                        range_cond: Dict[str, Any] = {}
                        if "$gte" in value or "gte" in value:
                            range_cond["gte"] = value.get("$gte", value.get("gte"))
                        if "$gt" in value or "gt" in value:
                            range_cond["gt"] = value.get("$gt", value.get("gt"))
                        if "$lte" in value or "lte" in value:
                            range_cond["lte"] = value.get("$lte", value.get("lte"))
                        if "$lt" in value or "lt" in value:
                            range_cond["lt"] = value.get("$lt", value.get("lt"))
                        if range_cond:
                            conditions_to_add.append({"key": key, "range": range_cond})
                else:
                    # Simple equality (type-tolerant: int/string both match)
                    conditions_to_add.append(_qdrant_match_condition(key, value))

    if not conditions_to_add:
        return None

    return {"must": conditions_to_add}

    merged = dict(filters)
    existing_must = merged.get("must", [])
    must_conditions = list(existing_must) if isinstance(existing_must, list) else (
        [existing_must] if existing_must else []
    )
    # Remove any existing isActive, is_current_version, or version_number conditions
    must_conditions = [
        c for c in must_conditions
        if not (isinstance(c, dict) and c.get("key") in ("isActive", "is_current_version", "version_number"))
    ]
    must_conditions.extend(conditions_to_add)
    merged["must"] = must_conditions
    return merged


def _normalize_user_filters(
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Normalize user-provided filters to a list of conditions.
    
    Supports multiple input formats:
    - Qdrant format: {"must": [{"key": "field", "match": {"value": "x"}}]}
    - Simple format: {"field": "value"} -> [{"key": "field", "match": {"value": "value"}}]
    - Range format: {"field": {"gte": 0, "lte": 10}}
    
    Returns a list of normalized condition dicts with "key" and operator.
    """
    if not filters:
        return []
    
    conditions = []
    
    # Qdrant format with must/should/must_not
    if "must" in filters or "should" in filters or "must_not" in filters:
        for clause_type in ["must", "should", "must_not"]:
            clause_values = filters.get(clause_type, [])
            if not isinstance(clause_values, list):
                clause_values = [clause_values] if clause_values else []
            for cond in clause_values:
                if isinstance(cond, dict) and "key" in cond:
                    cond_copy = dict(cond)
                    cond_copy["_clause_type"] = clause_type
                    conditions.append(cond_copy)
        return conditions
    
    # Simple key-value format
    for key, value in filters.items():
        if isinstance(value, dict):
            # Range or operator format: {"gte": 0, "lte": 10} or {"$eq": "value"}
            conditions.append({"key": key, "match": value})
        else:
            # Simple equality: {"field": "value"}
            conditions.append({"key": key, "match": {"value": value}})
    
    return conditions


def _validate_filters(
    filters: Optional[Dict[str, Any]] = None,
    backend: str = "all",
) -> List[str]:
    """
    Validate filter structure for the specified backend.
    
    Returns a list of validation error messages (empty if valid).
    
    Supported operators per backend:
    - Qdrant: match.value, range.gte/lte/gt/lt
    - Pinecone: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, $and, $or
    - PGVector: All operators supported via JSONB queries
    - MongoDB: All MongoDB operators ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin, etc.)
    - Redis: Equality, $in, numeric ranges (limited support)
    """
    errors = []
    
    if not filters:
        return errors
    
    def _validate_condition(cond: Dict[str, Any], path: str = "root") -> None:
        """Recursively validate a single condition."""
        if not isinstance(cond, dict):
            errors.append(f"{path}: condition must be a dict, got {type(cond).__name__}")
            return
        
        # Qdrant-style condition
        if "key" in cond:
            key = cond.get("key")
            if not key or not isinstance(key, str):
                errors.append(f"{path}: 'key' must be a non-empty string")
            
            match = cond.get("match", {})
            if not isinstance(match, dict):
                errors.append(f"{path}: 'match' must be a dict")
                return
            
            # Check for valid match operators
            valid_ops = {"value", "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "gte", "gt", "lte", "lt"}
            for op in match.keys():
                if op not in valid_ops:
                    errors.append(f"{path}: unknown operator '{op}' in match")
        
        # Pinecone/MongoDB-style nested conditions
        elif "$and" in cond or "$or" in cond:
            for op in ["$and", "$or"]:
                if op in cond:
                    nested = cond[op]
                    if not isinstance(nested, list):
                        errors.append(f"{path}: '{op}' must be a list")
                    else:
                        for i, subcond in enumerate(nested):
                            _validate_condition(subcond, f"{path}.{op}[{i}]")
        
        # Simple field: value format or operator format
        elif len(cond) >= 1:
            for key, value in cond.items():
                if key.startswith("$"):
                    errors.append(f"{path}: unexpected operator '{key}' at field level")
                elif isinstance(value, dict):
                    # Check for unknown operators in the value
                    valid_ops = {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "value", "gte", "gt", "lte", "lt"}
                    for op in value.keys():
                        if op not in valid_ops:
                            errors.append(f"{path}.{key}: unknown operator '{op}'")
        
        # Multiple fields at once (valid for some backends)
        else:
            if backend == "redis":
                errors.append(
                    f"{path}: Redis only supports single-field conditions; "
                    "use multiple conditions in a single filter dict"
                )
    
    # Validate top-level structure
    if "must" in filters or "should" in filters or "must_not" in filters:
        # Qdrant format
        for clause_type in ["must", "should", "must_not"]:
            clause = filters.get(clause_type)
            if clause is not None:
                if not isinstance(clause, list):
                    clause = [clause]
                for i, cond in enumerate(clause):
                    _validate_condition(cond, f"{clause_type}[{i}]")
    else:
        # Simple format or Pinecone format
        _validate_condition(filters, "filters")
    
    return errors


def _build_pinecone_filter(
    filters: Optional[Dict[str, Any]] = None,
    is_active: Optional[bool] = True,
    version: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build Pinecone metadata filter from user filters and isActive.
    
    Pinecone filter syntax:
    {"field": {"$eq": "value"}}
    {"field": {"$in": ["a", "b"]}}
    {"$and": [{"field1": {"$eq": "x"}}, {"field2": {"$eq": "y"}}]}
    """
    conditions = _normalize_user_filters(filters)
    pinecone_conditions = []
    
    # Add isActive filter (source-level)
    if is_active is not None:
        pinecone_conditions.append({"isActive": {"$eq": is_active}})
    
    # Version filter: specific version or current only
    if version is not None:
        pinecone_conditions.append({"version_number": {"$eq": version}})
    else:
        pinecone_conditions.append({"is_current_version": {"$eq": True}})
    
    # Convert normalized conditions to Pinecone format
    for cond in conditions:
        key = cond.get("key")
        match = cond.get("match", {})
        
        if not key:
            continue
        
        # Build range condition for combined operators
        range_cond = {}
        has_range = False
        for op in ["$gte", "$gt", "$lte", "$lt", "gte", "gt", "lte", "lt"]:
            if op in match:
                # Normalize operator names (remove $ prefix for consistent handling)
                norm_op = op if op.startswith("$") else f"${op}"
                range_cond[norm_op] = match[op]
                has_range = True
        
        # Handle different match formats
        if "value" in match:
            # Simple equality
            pinecone_conditions.append({key: {"$eq": match["value"]}})
        elif "$eq" in match:
            pinecone_conditions.append({key: {"$eq": match["$eq"]}})
        elif "$ne" in match:
            pinecone_conditions.append({key: {"$ne": match["$ne"]}})
        elif "$in" in match:
            pinecone_conditions.append({key: {"$in": match["$in"]}})
        elif "$nin" in match:
            pinecone_conditions.append({key: {"$nin": match["$nin"]}})
        elif has_range:
            # Range condition (combined $gte/$lte etc.)
            pinecone_conditions.append({key: range_cond})
        elif "$gt" in match:
            pinecone_conditions.append({key: {"$gt": match["$gt"]}})
        elif "$gte" in match:
            pinecone_conditions.append({key: {"$gte": match["$gte"]}})
        elif "$lt" in match:
            pinecone_conditions.append({key: {"$lt": match["$lt"]}})
        elif "$lte" in match:
            pinecone_conditions.append({key: {"$lte": match["$lte"]}})
    
    # Combine conditions
    if not pinecone_conditions:
        return None
    if len(pinecone_conditions) == 1:
        return pinecone_conditions[0]
    return {"$and": pinecone_conditions}


def _build_pgvector_where_clause(
    filters: Optional[Dict[str, Any]] = None,
    is_active: Optional[bool] = True,
    version: Optional[int] = None,
) -> Tuple[List[str], List[Any]]:
    """
    Build PGVector WHERE clauses from user filters and isActive.
    Returns (where_clauses, params) tuple.
    
    Filters are applied against the metadata JSONB column.
    """
    conditions = _normalize_user_filters(filters)
    where_clauses = []
    params = []
    
    # Add isActive filter (source-level)
    if is_active is not None:
        where_clauses.append("COALESCE((metadata->>'isActive')::boolean, true) = %s")
        params.append(is_active)
    
    # Version filter: specific version or current only
    if version is not None:
        where_clauses.append("(metadata->>'version_number')::int = %s")
        params.append(version)
    else:
        where_clauses.append("COALESCE((metadata->>'is_current_version')::boolean, true) = %s")
        params.append(True)
    
    # Convert normalized conditions to SQL
    for cond in conditions:
        key = cond.get("key")
        match = cond.get("match", {})
        
        if not key:
            continue
        
        # Sanitize key for SQL (only allow alphanumeric and underscore)
        safe_key = re.sub(r"[^a-zA-Z0-9_]", "_", str(key))
        
        # Handle different match formats
        if "value" in match:
            where_clauses.append(f"metadata->>'{safe_key}' = %s")
            params.append(str(match["value"]))
        elif "$eq" in match:
            where_clauses.append(f"metadata->>'{safe_key}' = %s")
            params.append(str(match["$eq"]))
        elif "$ne" in match:
            where_clauses.append(f"metadata->>'{safe_key}' != %s")
            params.append(str(match["$ne"]))
        elif "$gt" in match:
            where_clauses.append(f"(metadata->>'{safe_key}')::float > %s")
            params.append(float(match["$gt"]))
        elif "$gte" in match:
            where_clauses.append(f"(metadata->>'{safe_key}')::float >= %s")
            params.append(float(match["$gte"]))
        elif "$lt" in match:
            where_clauses.append(f"(metadata->>'{safe_key}')::float < %s")
            params.append(float(match["$lt"]))
        elif "$lte" in match:
            where_clauses.append(f"(metadata->>'{safe_key}')::float <= %s")
            params.append(float(match["$lte"]))
        elif "$in" in match:
            values = match["$in"]
            placeholders = ", ".join(["%s"] * len(values))
            where_clauses.append(f"metadata->>'{safe_key}' IN ({placeholders})")
            params.extend([str(v) for v in values])
        elif any(op in match for op in ("gte", "gt", "lte", "lt")):
            # Range format
            if "gte" in match:
                where_clauses.append(f"(metadata->>'{safe_key}')::float >= %s")
                params.append(float(match["gte"]))
            if "gt" in match:
                where_clauses.append(f"(metadata->>'{safe_key}')::float > %s")
                params.append(float(match["gt"]))
            if "lte" in match:
                where_clauses.append(f"(metadata->>'{safe_key}')::float <= %s")
                params.append(float(match["lte"]))
            if "lt" in match:
                where_clauses.append(f"(metadata->>'{safe_key}')::float < %s")
                params.append(float(match["lt"]))
    
    return where_clauses, params


def _build_mongo_filter(
    filters: Optional[Dict[str, Any]] = None,
    is_active: Optional[bool] = True,
    version: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build MongoDB query filter from user filters and isActive.
    """
    conditions = _normalize_user_filters(filters)
    mongo_filter = {}
    
    # Add isActive filter (source-level)
    if is_active is not None:
        mongo_filter["isActive"] = is_active
    
    # Version filter: specific version or current only
    if version is not None:
        mongo_filter["version_number"] = version
    else:
        mongo_filter["is_current_version"] = True
    
    # Convert normalized conditions to MongoDB format
    for cond in conditions:
        key = cond.get("key")
        match = cond.get("match", {})
        
        if not key:
            continue
        
        # Build range condition for combined operators
        range_cond = {}
        has_range = False
        for op in ["$gte", "$gt", "$lte", "$lt", "gte", "gt", "lte", "lt"]:
            if op in match:
                # Normalize operator names (remove $ prefix for consistent handling)
                norm_op = op if op.startswith("$") else f"${op}"
                range_cond[norm_op] = match[op]
                has_range = True
        
        # Handle different match formats
        if "value" in match:
            mongo_filter[key] = match["value"]
        elif "$eq" in match:
            mongo_filter[key] = match["$eq"]
        elif "$ne" in match:
            mongo_filter[key] = {"$ne": match["$ne"]}
        elif "$in" in match:
            mongo_filter[key] = {"$in": match["$in"]}
        elif "$nin" in match:
            mongo_filter[key] = {"$nin": match["$nin"]}
        elif has_range:
            # Range condition (combined $gte/$lte etc.)
            mongo_filter[key] = range_cond
        elif "$gt" in match:
            mongo_filter[key] = {"$gt": match["$gt"]}
        elif "$gte" in match:
            mongo_filter[key] = {"$gte": match["$gte"]}
        elif "$lt" in match:
            mongo_filter[key] = {"$lt": match["$lt"]}
        elif "$lte" in match:
            mongo_filter[key] = {"$lte": match["$lte"]}
    
    return mongo_filter


def _build_redis_filter(
    filters: Optional[Dict[str, Any]] = None,
    is_active: Optional[bool] = True,
    version: Optional[int] = None,
) -> Optional[str]:
    """
    Build RediSearch filter query string from user filters and isActive.
    Returns a filter string for FT.SEARCH.
    
    RediSearch syntax: @field:value, @field:[min max], etc.
    """
    conditions = _normalize_user_filters(filters)
    redis_conditions = []
    
    # Add isActive filter (source-level, stored as tag field)
    if is_active is not None:
        redis_conditions.append(f"@isActive:{{{str(is_active).lower()}}}")
    
    # Version filter: specific version or current only
    if version is not None:
        redis_conditions.append(f"@version_number:[{version} {version}]")
    else:
        redis_conditions.append("@is_current_version:{true}")
    
    # Convert normalized conditions to RediSearch format
    for cond in conditions:
        key = cond.get("key")
        match = cond.get("match", {})
        
        if not key:
            continue
        
        # Handle different match formats
        if "value" in match:
            # Exact match on tag field
            val = str(match["value"]).replace("'", "\\'")
            redis_conditions.append(f"@{key}:{{{val}}}")
        elif "$eq" in match:
            val = str(match["$eq"]).replace("'", "\\'")
            redis_conditions.append(f"@{key}:{{{val}}}")
        elif "$in" in match:
            values = [str(v).replace("'", "\\'") for v in match["$in"]]
            redis_conditions.append(f"@{key}:{{{'|'.join(values)}}}")
        elif any(op in match for op in ("gte", "gt", "lte", "lt")):
            # Numeric range
            min_val = match.get("gte", match.get("gt", "-inf"))
            max_val = match.get("lte", match.get("lt", "+inf"))
            redis_conditions.append(f"@{key}:[{min_val} {max_val}]")
        elif "$gt" in match or "$lt" in match:
            min_val = match.get("$gt", match.get("$gte", "-inf"))
            max_val = match.get("$lt", match.get("$lte", "+inf"))
            redis_conditions.append(f"@{key}:[{min_val} {max_val}]")
    
    if not redis_conditions:
        return None
    return " ".join(redis_conditions)



def _qdrant_headers(api_key: Optional[str]) -> dict:
    return {"api-key": api_key} if api_key else {}


def _qdrant_collection_info(
    collection: str,
    base_url: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch collection schema — used to detect vector names and server version."""
    url     = f"{base_url}/collections/{collection}"
    headers = _qdrant_headers(api_key)
    safe_headers = {k: ("***" if k.lower() == "api-key" else v) for k, v in headers.items()}
    print(
        f"  [qdrant-debug] GET {url} | headers={safe_headers}",
        flush=True,
    )
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        print(
            f"  [qdrant-debug] collection info response: status={resp.status_code} "
            f"body={resp.text[:300]}",
            flush=True,
        )
        if resp.status_code == 200:
            result = resp.json().get("result", {})
            config = result.get("config", {})
            params = config.get("params", {})
            print(
                f"  [qdrant-debug] vectors={list(params.get('vectors', {}).keys())} "
                f"sparse_vectors={list(params.get('sparse_vectors', {}).keys())}",
                flush=True,
            )
            return result
        else:
            logger.warning(
                "Qdrant collection info %d: %s", resp.status_code, resp.text[:200]
            )
    except Exception as exc:
        logger.warning("Qdrant collection info failed: %s", exc)
    return {}


def _qdrant_detect_vector_names(
    collection: str,
    base_url: str,
    api_key: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Return (dense_vector_name, sparse_vector_name) as actually stored in
    the collection schema.

    Falls back to the global constants if the schema cannot be fetched or
    does not contain named vectors.
    """
    info   = _qdrant_collection_info(collection, base_url, api_key)
    config = info.get("config", {})
    params = config.get("params", {})

    # Named dense vectors live under params.vectors (dict keyed by name)
    vectors_cfg = params.get("vectors", {})
    # Named sparse vectors live under params.sparse_vectors (dict keyed by name)
    sparse_cfg  = params.get("sparse_vectors", {})

    dense_name  = DENSE_VECTOR_NAME
    sparse_name = SPARSE_VECTOR_NAME

    if isinstance(vectors_cfg, dict) and vectors_cfg:
        # prefer "dense" if present, else take the first key
        if DENSE_VECTOR_NAME in vectors_cfg:
            dense_name = DENSE_VECTOR_NAME
        else:
            dense_name = next(iter(vectors_cfg))

    if isinstance(sparse_cfg, dict) and sparse_cfg:
        if SPARSE_VECTOR_NAME in sparse_cfg:
            sparse_name = SPARSE_VECTOR_NAME
        else:
            sparse_name = next(iter(sparse_cfg))

    return dense_name, sparse_name


def _qdrant_log_request(url: str, headers: dict, payload: dict) -> None:
    """Log the outgoing Qdrant request (masks the API key value)."""
    safe_headers = {
        k: ("***" if k.lower() == "api-key" else v)
        for k, v in headers.items()
    }
    logger.debug(
        "Qdrant request | url=%s | headers=%s | payload_keys=%s",
        url, safe_headers, list(payload.keys()),
    )
    # Also print at INFO so it always shows in console during debugging
    print(
        f"  [qdrant-debug] POST {url} | headers={safe_headers} | "
        f"payload={list(payload.keys())}",
        flush=True,
    )

def _qdrant_ensure_is_active_index(
    base_url: str,
    collection: str,
    headers: dict,
) -> bool:
    """
    Create a boolean payload index on 'isActive' in the given Qdrant collection.

    Qdrant Cloud requires an explicit payload index before a boolean field can
    be used in a filter. This is called automatically whenever a 400
    "Index required" error is detected so the retry succeeds with the filter
    intact — isActive filtering is never silently dropped.

    Tries both known endpoint variants:
      PUT /collections/{name}/index           (Qdrant v1.x / Cloud)
      PUT /collections/{name}/payload/index   (older Qdrant builds)
    Returns True if the index was created or already existed, False on error.
    """
    body = {"field_name": "isActive", "field_schema": "bool"}
    endpoints = [
        f"{base_url}/collections/{collection}/index",
        f"{base_url}/collections/{collection}/payload/index",
    ]
    for url in endpoints:
        print(
            f"  [qdrant-debug] Creating isActive payload index — PUT {url}",
            flush=True,
        )
        try:
            resp = requests.put(url, json=body, headers=headers, timeout=15)
            if resp.ok:
                logger.info(
                    "Qdrant: created payload index on 'isActive' | collection=%s | url=%s",
                    collection, url,
                )
                print(
                    f"  [qdrant-debug] isActive index created successfully via {url}",
                    flush=True,
                )
                return True
            elif resp.status_code == 404:
                # This endpoint variant doesn't exist — try the next one
                logger.debug("Qdrant index endpoint not found: %s", url)
                continue
            else:
                logger.warning(
                    "Qdrant: failed to create isActive index via %s (%d): %s",
                    url, resp.status_code, resp.text[:200],
                )
                # Non-404 failure — still try next variant
                continue
        except Exception as exc:
            logger.warning("Qdrant: error calling %s: %s", url, exc)
            continue

    logger.error(
        "Qdrant: could not create isActive payload index for collection '%s' "
        "via any known endpoint. isActive filter may not work.",
        collection,
    )
    return False


# Maps the type names Qdrant reports in "Index required ... of the following
# types: [X]" to the field_schema value used when creating the index.
_QDRANT_INDEX_SCHEMA_BY_TYPE = {
    "integer": "integer",
    "float": "float",
    "keyword": "keyword",
    "bool": "bool",
    "boolean": "bool",
    "geo": "geo",
    "datetime": "datetime",
    "text": "text",
    "uuid": "uuid",
}


def _parse_missing_index_error(body: str) -> Optional[Tuple[str, str]]:
    """Extract (field_name, field_schema) from a Qdrant 'Index required' error.

    Example error text:
      Index required but not found for "org_id" of one of the following types: [integer]

    Returns (field_name, field_schema) or None when it can't be parsed.
    """
    import re

    # The field name is quoted in the error, but in the raw JSON body the
    # quotes are often escaped (\"org_id\"). Match the name whether the
    # surrounding quote is a plain " or an escaped \" , and as a bare word.
    m = (
        re.search(r'not found for\s+\\?"([^"\\]+)\\?"', body)
        or re.search(r"not found for\s+'([^']+)'", body)
        or re.search(r"not found for\s+([A-Za-z_][\w]*)", body)
    )
    if not m:
        return None
    field_name = m.group(1)

    types_match = re.search(r"following types:\s*\[([^\]]*)\]", body)
    field_schema = "keyword"  # safe default for string-like fields
    if types_match:
        first_type = types_match.group(1).split(",")[0].strip().lower()
        field_schema = _QDRANT_INDEX_SCHEMA_BY_TYPE.get(first_type, "keyword")
    return field_name, field_schema


def _qdrant_ensure_payload_index(
    base_url: str,
    collection: str,
    headers: dict,
    field_name: str,
    field_schema: str,
) -> bool:
    """Create a payload index on an arbitrary field so it can be filtered.

    Qdrant requires an explicit payload index before a field can be used in a
    filter. Called automatically when a 400 'Index required' error names a
    field (e.g. org_id). Idempotent — an existing index is treated as success.
    """
    body = {"field_name": field_name, "field_schema": field_schema}
    endpoints = [
        f"{base_url}/collections/{collection}/index",
        f"{base_url}/collections/{collection}/payload/index",
    ]
    for url in endpoints:
        try:
            resp = requests.put(url, json=body, headers=headers, timeout=15)
            if resp.ok or "already exists" in resp.text.lower():
                logger.info(
                    "Qdrant: ensured payload index | collection=%s field=%s schema=%s",
                    collection, field_name, field_schema,
                )
                return True
            if resp.status_code == 404:
                continue  # endpoint variant not available — try the other

            # A different-typed index on this field can block creation
            # (ingestion sometimes creates a keyword index while values are
            # stored as integers). Drop the conflicting index and recreate it
            # with the type Qdrant actually needs, so filtering self-heals.
            lowered = resp.text.lower()
            if resp.status_code in (400, 409) and (
                "already exists" in lowered or "different" in lowered or "conflict" in lowered
                or "wrong" in lowered or "type" in lowered
            ):
                try:
                    del_url = f"{base_url}/collections/{collection}/index/{field_name}?wait=true"
                    drop = requests.delete(del_url, headers=headers, timeout=15)
                    logger.warning(
                        "Qdrant: dropping conflicting index on '%s' (status %s) then recreating as %s.",
                        field_name, drop.status_code, field_schema,
                    )
                    recreate = requests.put(url, json=body, headers=headers, timeout=15)
                    if recreate.ok:
                        return True
                    logger.warning(
                        "Qdrant: recreate index on '%s' failed (%d): %s",
                        field_name, recreate.status_code, recreate.text[:200],
                    )
                except Exception as exc:
                    logger.warning("Qdrant: error recreating index on %s: %s", field_name, exc)

            logger.warning(
                "Qdrant: failed to create index on %s via %s (%d): %s",
                field_name, url, resp.status_code, resp.text[:200],
            )
        except Exception as exc:
            logger.warning("Qdrant: error creating index on %s via %s: %s", field_name, url, exc)
    return False


def _qdrant_search_with_query_api(
    base_url: str,
    collection: str,
    headers: dict,
    payload: dict,
) -> List[RetrievalResult]:
    """
    Use the Qdrant v1.10+ /points/query endpoint.

    If the filter causes a 400 "Index required" error, automatically creates
    the payload index on 'isActive' and retries — ensuring isActive filtering
    always works without manual index setup.
    """
    url = f"{base_url}/collections/{collection}/points/query"
    _qdrant_log_request(url, headers, payload)
    resp = requests.post(url, json=payload, headers=headers, timeout=30)

    # ── Auto-create missing payload indexes + retry on "Index required" 400 ──
    # A single query may reference several unindexed fields (isActive,
    # is_current_version, org_id, ...). Qdrant reports them one at a time, so
    # loop: create the named index, retry, repeat until it stops complaining.
    attempts = 0
    while resp.status_code == 400 and attempts < 6:
        body = resp.text
        if "index required" not in body.lower() and "payload index" not in body.lower():
            break
        parsed = _parse_missing_index_error(body)
        if not parsed:
            # Fall back to the legacy isActive-only behavior.
            if not _qdrant_ensure_is_active_index(base_url, collection, headers):
                break
        else:
            field_name, field_schema = parsed
            logger.warning(
                "Qdrant: payload index missing for '%s' — creating (%s) and retrying.",
                field_name, field_schema,
            )
            if not _qdrant_ensure_payload_index(
                base_url, collection, headers, field_name, field_schema
            ):
                break
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        attempts += 1

    if not resp.ok:
        logger.error("Qdrant /query error %d: %s", resp.status_code, resp.text[:300])
    resp.raise_for_status()

    data   = resp.json()
    points = data.get("result", {})
    if isinstance(points, dict):
        points = points.get("points", [])
    return _parse_qdrant_hits(points)


def _qdrant_search_with_search_api(
    base_url: str,
    collection: str,
    headers: dict,
    payload: dict,
) -> List[RetrievalResult]:
    """
    Use the legacy Qdrant /points/search endpoint (pre-v1.10 / internal).

    If the filter causes a 400 "Index required" error, automatically creates
    the payload index on 'isActive' and retries.
    """
    url = f"{base_url}/collections/{collection}/points/search"
    _qdrant_log_request(url, headers, payload)
    resp = requests.post(url, json=payload, headers=headers, timeout=30)

    # ── Auto-create index + retry on "Index required" 400 ─────────────────
    if resp.status_code == 400:
        body = resp.text
        if "index required" in body.lower() or "payload index" in body.lower():
            logger.warning(
                "Qdrant isActive index missing — creating it automatically. Error: %s",
                body[:200],
            )
            created = _qdrant_ensure_is_active_index(base_url, collection, headers)
            if created:
                # Retry the original request with filter intact
                resp = requests.post(url, json=payload, headers=headers, timeout=30)
            else:
                logger.error(
                    "Could not create isActive index — search with isActive filter will fail"
                )

    if not resp.ok:
        logger.error("Qdrant /search error %d: %s", resp.status_code, resp.text[:300])
    resp.raise_for_status()
    return _parse_qdrant_hits(resp.json()["result"])


def _qdrant_search_dense(
    vector: List[float],
    collection: str,
    base_url: str,
    *,
    limit: int = 5,
    score_threshold: float = 0.35,
    filters: Optional[dict] = None,
    is_active: Optional[bool] = True,
    version: Optional[int] = None,
    api_key: Optional[str] = None,
) -> List[RetrievalResult]:
    """
    Search dense vectors on user-provided Qdrant (Cloud / v1.10+).
    Uses /points/query exclusively — the correct endpoint for Qdrant Cloud.
    No fallback to /points/search to avoid masking real errors.
    """
    headers       = _qdrant_headers(api_key)
    qdrant_filter = _apply_is_active_filter(filters, is_active, version)
    dense_name, _ = _qdrant_detect_vector_names(collection, base_url, api_key)

    payload: dict = {
        "query":           {"nearest": vector},   # /points/query requires wrapped format
        "using":           dense_name,
        "limit":           limit,
        "with_payload":    True,
        "score_threshold": score_threshold,
    }
    if qdrant_filter is not None:
        payload["filter"] = qdrant_filter

    return _qdrant_search_with_query_api(base_url, collection, headers, payload)


def _qdrant_search_sparse(
    sparse_vec: Dict[str, List],
    collection: str,
    base_url: str,
    *,
    limit: int = 5,
    filters: Optional[dict] = None,
    is_active: Optional[bool] = True,
    version: Optional[int] = None,
    api_key: Optional[str] = None,
) -> List[RetrievalResult]:
    """
    Search sparse vectors on user-provided Qdrant (Cloud / v1.10+).
    Uses /points/query exclusively — no fallback to /points/search.
    """
    headers        = _qdrant_headers(api_key)
    qdrant_filter  = _apply_is_active_filter(filters, is_active, version)
    _, sparse_name = _qdrant_detect_vector_names(collection, base_url, api_key)

    payload: dict = {
        "query": {
            "nearest": {
                "indices": sparse_vec["indices"],
                "values":  sparse_vec["values"],
            }
        },
        "using":        sparse_name,
        "limit":        limit,
        "with_payload": True,
    }
    if qdrant_filter is not None:
        payload["filter"] = qdrant_filter

    return _qdrant_search_with_query_api(base_url, collection, headers, payload)


# ========================= PINECONE BACKEND =========================

def _pinecone_search(
    query_vector: List[float],
    vs_details: Dict[str, Any],
    collection: str,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    sparse_vec: Optional[Dict[str, List]] = None,
    is_active: Optional[bool] = True,
    filters: Optional[Dict[str, Any]] = None,
    version: Optional[int] = None,
) -> List[RetrievalResult]:
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise ImportError("pip install pinecone-client") from exc

    api_key    = vs_details.get("PINECONE_API_KEY") or vs_details.get("pinecone_api_key")
    index_name = vs_details.get("PINECONE_INDEX_NAME") or vs_details.get("pinecone_index_name")
    namespace  = vs_details.get("PINECONE_NAMESPACE") or vs_details.get("pinecone_namespace") or collection

    pc    = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    query_kwargs: Dict[str, Any] = {
        "namespace":        namespace,
        "top_k":            limit,
        "include_metadata": True,
    }
    # Only include the dense vector when it is non-empty.
    # Passing vector=[] to a dense index causes a 400 "Cannot query index
    # with dense vector_type with only sparse vector" error.
    if query_vector:
        query_kwargs["vector"] = query_vector
    elif not sparse_vec:
        raise ValueError(
            "Pinecone search requires at least one of: a dense query vector "
            "or a sparse vector. Both are empty."
        )

    if sparse_vec:
        _validate_pinecone_sparse_query_support(pc, index_name)
        query_kwargs["sparse_vector"] = {
            "indices": sparse_vec["indices"],
            "values":  sparse_vec["values"],
        }

    # Build metadata filter (isActive + user filters)
    pinecone_filter = _build_pinecone_filter(filters, is_active, version)
    if pinecone_filter:
        query_kwargs["filter"] = pinecone_filter

    response = index.query(**query_kwargs)
    matches  = (
        response.get("matches", []) if isinstance(response, dict)
        else getattr(response, "matches", [])
    )

    results: List[RetrievalResult] = []
    for m in matches:
        if isinstance(m, dict):
            score = float(m.get("score", 0.0))
            meta  = _normalize_pinecone_metadata(m.get("metadata") or {})
            pid   = str(m.get("id", ""))
        else:
            score = float(getattr(m, "score", 0.0))
            meta  = _normalize_pinecone_metadata(getattr(m, "metadata", {}) or {})
            pid   = str(getattr(m, "id", ""))

        if score < score_threshold:
            continue
        results.append(RetrievalResult(
            score    = score,
            text     = meta.get("text", ""),
            metadata = meta,
            point_id = pid,
        ))
    return results


# ========================= PGVECTOR BACKEND =========================

def _pgvector_connect(vs_details: Dict[str, Any]):
    """
    Return a psycopg2 connection to the target PGVector database.

    If the database does not exist (error code 3D000) it is created
    automatically by connecting to the 'postgres' maintenance database first,
    mirroring the same logic in PGVectorStore._ensure_database().
    """
    try:
        import psycopg2
    except ImportError as exc:
        raise ImportError("pip install psycopg2-binary") from exc

    host     = vs_details.get("PG_HOST", "localhost")
    port     = int(vs_details.get("PG_PORT", 5432))
    database = vs_details.get("PG_DATABASE", "postgres")
    user     = vs_details.get("PG_USER", "postgres")
    password = vs_details.get("PG_PASSWORD", "")

    def _connect():
        return psycopg2.connect(
            host=host, port=port, dbname=database,
            user=user, password=password, connect_timeout=15,
        )

    try:
        return _connect()
    except psycopg2.OperationalError as exc:
        pg_code = getattr(exc, "pgcode", None) or ""
        if pg_code == "3D000" or "does not exist" in str(exc):
            # Database missing — create it via the maintenance DB then retry
            admin = psycopg2.connect(
                host=host, port=port, dbname="postgres",
                user=user, password=password, connect_timeout=15,
            )
            admin.autocommit = True
            with admin.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s;", (database,)
                )
                if not cur.fetchone():
                    cur.execute('CREATE DATABASE "%s";' % database.replace('"', '""'))
                    logger.info("Created database '%s' for retrieval", database)
            admin.close()
            return _connect()
        raise


def _pgvector_search(
    query_vector: List[float],
    vs_details: Dict[str, Any],
    collection: str,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    is_active: Optional[bool] = True,
    filters: Optional[Dict[str, Any]] = None,
    version: Optional[int] = None,
) -> List[RetrievalResult]:
    import re
    try:
        import psycopg2.extras
    except ImportError as exc:
        raise ImportError("pip install psycopg2-binary") from exc

    table = re.sub(r"[^a-zA-Z0-9_]", "_", collection)

    conn = _pgvector_connect(vs_details)

    # Build WHERE clause from filters and isActive
    where_clauses, filter_params = _build_pgvector_where_clause(filters, is_active, version)
    params: List[Any] = [query_vector] + filter_params

    where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    # Cosine distance: 1 - cosine_similarity → similarity = 1 - distance
    sql = f"""
        SELECT
            chunk_id,
            text,
            metadata,
            chunk_index,
            1 - (embedding <=> %s::vector)  AS score
        FROM "{table.replace(chr(34), chr(34) * 2)}"
        {where_sql}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    params.append(query_vector)   # second use for ORDER BY
    params.append(limit)

    results: List[RetrievalResult] = []
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            for row in cur.fetchall():
                score = float(row["score"])
                if score < score_threshold:
                    continue
                meta = row["metadata"] or {}
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}
                results.append(RetrievalResult(
                    score    = score,
                    text     = row.get("text") or "",
                    metadata = meta,
                    point_id = str(row.get("chunk_id", "")),
                ))
    finally:
        conn.close()

    return results


# ========================= MONGODB ATLAS BACKEND =========================

def _create_mongo_atlas_client(
    uri: str,
    *,
    vs_details: Optional[Dict[str, Any]] = None,
    server_selection_timeout_ms: int = 10_000,
    connect_timeout_ms: int = 10_000,
    socket_timeout_ms: int = 30_000,
):
    try:
        from pymongo import MongoClient
    except ImportError as exc:
        raise ImportError("pip install pymongo") from exc

    client_kwargs: Dict[str, Any] = {
        "serverSelectionTimeoutMS": server_selection_timeout_ms,
        "connectTimeoutMS": connect_timeout_ms,
        "socketTimeoutMS": socket_timeout_ms,
    }
    vs_details = vs_details or {}
    normalized_uri = (uri or "").lower()

    def _cfg(*keys: str) -> Any:
        for key in keys:
            value = vs_details.get(key)
            if value not in (None, ""):
                return value
        return None

    def _parse_optional_bool(value: Any) -> Optional[bool]:
        if value in (None, ""):
            return None
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Invalid boolean value for MongoDB TLS option: {value!r}")

    # Atlas requires TLS, but let explicit URI options win when present.
    explicit_tls = _parse_optional_bool(_cfg("MONGODB_TLS", "mongodb_tls"))
    if explicit_tls is not None:
        client_kwargs["tls"] = explicit_tls
    elif "tls=" not in normalized_uri and "ssl=" not in normalized_uri:
        client_kwargs["tls"] = True

    tls_insecure = _parse_optional_bool(
        _cfg("MONGODB_TLS_INSECURE", "mongodb_tls_insecure")
    )
    if tls_insecure is not None:
        client_kwargs["tlsInsecure"] = tls_insecure

    tls_allow_invalid_certificates = _parse_optional_bool(
        _cfg(
            "MONGODB_TLS_ALLOW_INVALID_CERTIFICATES",
            "mongodb_tls_allow_invalid_certificates",
        )
    )
    if tls_allow_invalid_certificates is not None:
        client_kwargs["tlsAllowInvalidCertificates"] = tls_allow_invalid_certificates

    tls_allow_invalid_hostnames = _parse_optional_bool(
        _cfg(
            "MONGODB_TLS_ALLOW_INVALID_HOSTNAMES",
            "mongodb_tls_allow_invalid_hostnames",
        )
    )
    if tls_allow_invalid_hostnames is not None:
        client_kwargs["tlsAllowInvalidHostnames"] = tls_allow_invalid_hostnames

    tls_ca_file = _cfg("MONGODB_TLS_CA_FILE", "mongodb_tls_ca_file")
    if tls_ca_file:
        client_kwargs["tlsCAFile"] = str(tls_ca_file)

    tls_certificate_key_file = _cfg(
        "MONGODB_TLS_CERTIFICATE_KEY_FILE",
        "mongodb_tls_certificate_key_file",
    )
    if tls_certificate_key_file:
        client_kwargs["tlsCertificateKeyFile"] = str(tls_certificate_key_file)

    tls_certificate_key_file_password = _cfg(
        "MONGODB_TLS_CERTIFICATE_KEY_FILE_PASSWORD",
        "mongodb_tls_certificate_key_file_password",
    )
    if tls_certificate_key_file_password:
        client_kwargs["tlsCertificateKeyFilePassword"] = str(
            tls_certificate_key_file_password
        )

    if normalized_uri.startswith("mongodb+srv://") or ".mongodb.net" in normalized_uri:
        if "tlsCAFile" not in client_kwargs:
            try:
                import certifi
                client_kwargs["tlsCAFile"] = certifi.where()
            except ImportError:
                pass  # certifi not installed — rely on system trust store

    return MongoClient(uri, **client_kwargs)

def _mongo_run_pipeline(client, database: str, collection: str, pipeline: list) -> list:
    """Execute an aggregation pipeline and return all documents."""
    col = client[database][collection]
    return list(col.aggregate(pipeline))


def _mongo_search(
    query_vector: List[float],
    vs_details: Dict[str, Any],
    collection: str,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    is_active: Optional[bool] = True,
    filters: Optional[Dict[str, Any]] = None,
    version: Optional[int] = None,
) -> List[RetrievalResult]:
    try:
        from pymongo.errors import (
            ServerSelectionTimeoutError,
            OperationFailure,
            ConfigurationError,
        )
    except ImportError as exc:
        raise ImportError("pip install pymongo") from exc

    uri         = vs_details.get("MONGODB_URI") or vs_details.get("mongodb_uri")
    database    = vs_details.get("MONGODB_DATABASE") or vs_details.get("mongodb_database")
    dense_field = vs_details.get("MONGODB_DENSE_FIELD") or vs_details.get("mongodb_dense_field") or "dense"

    # Guard: if the caller mistakenly set MONGODB_DENSE_FIELD to the sparse
    # field name (e.g. "sparse"), fall back to "dense".  The sparse field stores
    # {indices, values} dicts — not float arrays — so $vectorSearch will always
    # reject it with "X is not indexed as vector".
    _sparse_field_name = (
        vs_details.get("MONGODB_SPARSE_FIELD")
        or vs_details.get("mongodb_sparse_field")
        or "sparse"
    )
    if dense_field == _sparse_field_name:
        logger.warning(
            "MongoDB: MONGODB_DENSE_FIELD='%s' matches the sparse field name — "
            "overriding to 'dense'. Pass MONGODB_DENSE_FIELD explicitly if your "
            "dense vectors are stored under a different field name.",
            dense_field,
        )
        dense_field = "dense"

    if not uri:
        raise ValueError("MongoDB backend requires 'MONGODB_URI' in vector_store_details.")
    if not database:
        raise ValueError("MongoDB backend requires 'MONGODB_DATABASE' in vector_store_details.")

    # Build MongoDB filter from user filters and isActive
    mongo_filter = _build_mongo_filter(filters, is_active, version)

    # ==================== INDEX RESOLUTION ====================
    # Priority:
    #   1. Explicit MONGODB_INDEX_NAME in vs_details  → use it directly, no probing
    #   2. list_search_indexes() discovery            → works on M10+ clusters only
    #   3. Probe common names via $vectorSearch        → works on ALL tiers (M0/M2/M5/M10+)
    #      Probed in order: "vector_index", "default", collection name
    # =========================================================

    _explicit_index = (
        vs_details.get("MONGODB_INDEX_NAME") or
        vs_details.get("mongodb_index_name")
    )

    def _make_pipeline(idx: str, mongo_filter: Optional[Dict[str, Any]] = None) -> list:
        vs: Dict[str, Any] = {
            "index":         idx,
            "path":          dense_field,
            "queryVector":   query_vector,
            "numCandidates": max(limit * 10, 100),
            "limit":         limit,
        }
        # Add filter if provided
        if mongo_filter:
            vs["filter"] = mongo_filter
        return [
            {"$vectorSearch": vs},
            {"$addFields": {"_score": {"$meta": "vectorSearchScore"}}},
        ]

    def _is_index_error(exc: Exception) -> bool:
        """Return True when the error indicates a bad/missing index name."""
        msg = str(exc).lower()
        return any(k in msg for k in (
            "index not found", "no such index", "invalid index",
            "unknown index", "index named", "does not exist",
        ))

    # ── Build a shared client for all attempts ──────────────────────────
    client = None
    docs: list = []
    last_exc: Optional[Exception] = None
    index_name: Optional[str] = None

    try:
        client = _create_mongo_atlas_client(uri, vs_details=vs_details)
    except Exception as _conn_exc:
        if _is_mongo_tls_error(str(_conn_exc)):
            logger.warning("MongoDB TLS error — retrying with relaxed TLS settings.")
            try:
                relaxed = dict(vs_details)
                relaxed["MONGODB_TLS_ALLOW_INVALID_CERTIFICATES"] = True
                relaxed["MONGODB_TLS_ALLOW_INVALID_HOSTNAMES"] = True
                client = _create_mongo_atlas_client(uri, vs_details=relaxed)
            except Exception as retry_conn:
                raise RuntimeError(
                    f"MongoDB Atlas TLS handshake failed. {retry_conn}"
                ) from retry_conn
        else:
            raise

    try:
        # ── Path 1: explicit index name supplied by caller ───────────────
        if _explicit_index:
            index_name = _explicit_index
            logger.info(
                "MongoDB: Using explicit index_name='%s' dense_field='%s'",
                index_name, dense_field,
            )
            pipeline = _make_pipeline(index_name, mongo_filter)
            try:
                docs = _mongo_run_pipeline(client, database, collection, pipeline)
            except Exception as exc:
                last_exc = exc

        else:
            # ── Path 2: list_search_indexes() — M10+ clusters only ───────
            discovered: Optional[str] = None
            try:
                col = client[database][collection]
                _indexes = list(col.list_search_indexes())
                if _indexes:
                    _idx_names = [i.get("name") for i in _indexes]
                    logger.info(
                        "MongoDB: Discovered search indexes on '%s': %s",
                        collection, _idx_names,
                    )
                    # Prefer index whose definition references the dense_field
                    for idx in _indexes:
                        idx_str = json.dumps(idx, default=str)
                        if dense_field in idx_str or "vector" in idx_str.lower():
                            discovered = idx.get("name")
                            break
                    if not discovered:
                        discovered = _indexes[0].get("name")
                    logger.info("MongoDB: Auto-selected index '%s'", discovered)
                else:
                    logger.info(
                        "MongoDB: list_search_indexes() returned nothing for '%s' "
                        "(M0/M2/M5 free tier does not support this command — "
                        "will probe common index names instead)", collection,
                    )
            except Exception as _disco_exc:
                logger.info(
                    "MongoDB: list_search_indexes() unavailable (%s) — "
                    "will probe common index names", _disco_exc,
                )

            # ── Path 3: probe common index names ─────────────────────────
            # Build candidate list: discovered name first, then well-known defaults
            _candidates: List[str] = []
            if discovered:
                _candidates.append(discovered)
            # Always include common defaults so we handle M0 clusters
            for _c in ("vector_index", "default", collection):
                if _c and _c not in _candidates:
                    _candidates.append(_c)

            for candidate in _candidates:
                pipeline = _make_pipeline(candidate, mongo_filter)
                try:
                    _probe_docs = _mongo_run_pipeline(client, database, collection, pipeline)
                    # $vectorSearch succeeded — use this index
                    index_name = candidate
                    docs = _probe_docs
                    logger.info(
                        "MongoDB: index '%s' works for collection '%s' "
                        "(dense_field='%s', docs=%d)",
                        index_name, collection, dense_field, len(docs),
                    )
                    last_exc = None
                    break
                except Exception as probe_exc:
                    logger.debug(
                        "MongoDB: probe index '%s' failed: %s", candidate, probe_exc
                    )
                    last_exc = probe_exc
                    # Continue to next candidate only for index-not-found errors;
                    # stop immediately for TLS / auth / network errors.
                    if not _is_index_error(probe_exc):
                        break

            if index_name is None:
                # None of the probes succeeded
                logger.error(
                    "MongoDB: Could not find a working Vector Search index on "
                    "collection '%s'. Tried: %s. "
                    "Please create a Vector Search index in the Atlas UI and pass "
                    "MONGODB_INDEX_NAME in vector_store_details.",
                    collection, _candidates,
                )

        # ── 0-result diagnostic ─────────────────────────────────────────
        if last_exc is None and not docs and index_name:
            try:
                total = client[database][collection].count_documents({})
                logger.warning(
                    "MongoDB $vectorSearch returned 0 results. "
                    "Collection '%s' has %d docs. "
                    "index='%s' path='%s' — verify the index is READY in Atlas UI "
                    "and the dense field contains float arrays.",
                    collection, total, index_name, dense_field,
                )
            except Exception:
                pass

    finally:
        if client:
            try:
                client.close()
            except Exception:
                pass

    if last_exc is not None:
        if _is_mongo_tls_error(str(last_exc)):
            raise RuntimeError(f"MongoDB Atlas TLS handshake failed. {last_exc}") from last_exc
        raise last_exc

    # === Result Processing (unchanged) ===
    def _mongo_is_active(doc: dict, is_active_filter: Optional[bool]) -> bool:
        if is_active_filter is None:
            return True
        val = doc.get("isActive")
        if val is None:
            payload_raw = doc.get("payload")
            if isinstance(payload_raw, str):
                try:
                    val = json.loads(payload_raw).get("isActive")
                except:
                    pass
            elif isinstance(payload_raw, dict):
                val = payload_raw.get("isActive")
        if val is None:
            return True
        if isinstance(val, bool):
            return val == is_active_filter
        return str(val).strip().lower() == ("true" if is_active_filter else "false")

    results: List[RetrievalResult] = []
    for doc in docs:
        score = float(doc.get("_score", 0.0))
        if score < score_threshold:
            continue
        if not _mongo_is_active(doc, is_active):
            continue

        text = str(doc.get("text") or "")
        # Exclude raw vector fields and internal score key from returned metadata
        _sparse_field = (
            vs_details.get("MONGODB_SPARSE_FIELD")
            or vs_details.get("mongodb_sparse_field")
            or "sparse"
        )
        _vector_keys = {"_id", "_score", dense_field, _sparse_field}
        meta = {k: v for k, v in doc.items() if k not in _vector_keys}
        
        results.append(RetrievalResult(
            score    = score,
            text     = text,
            metadata = meta,
            point_id = str(doc.get("chunk_id") or doc.get("_id", "")),
        ))

    return results


# ========================= REDIS BACKEND =========================

def _parse_redis_url(url: str) -> Dict[str, Any]:
    """
    Parse a Redis connection URL into its components.

    Supported schemes:
      redis://[user:password@]host[:port][/db]
      rediss://[user:password@]host[:port][/db]   (TLS)

    Returns a dict with keys: host, port, password, username, db, ssl.
    """
    from urllib.parse import urlparse, unquote

    parsed = urlparse(url)
    scheme = (parsed.scheme or "redis").lower()
    host   = parsed.hostname or "localhost"
    port   = parsed.port or (6380 if scheme == "rediss" else 6379)
    ssl    = scheme == "rediss"

    username = unquote(parsed.username) if parsed.username else "default"
    password = unquote(parsed.password) if parsed.password else None

    # Path component: /0, /1 … encodes the DB number
    path = (parsed.path or "").lstrip("/")
    try:
        db = int(path) if path else 0
    except ValueError:
        db = 0

    return {
        "host":     host,
        "port":     port,
        "password": password,
        "username": username,
        "db":       db,
        "ssl":      ssl,
    }


def _redis_connect(vs_details: Dict[str, Any]):
    """
    Build a Redis client from *either*:
      • A URL  — vs_details["redis_url"] / vs_details["REDIS_URL"]
      • Individual fields — REDIS_HOST, REDIS_PORT, REDIS_PASSWORD, …

    The URL takes precedence when both are present.
    """
    try:
        import redis as redis_lib
    except ImportError as exc:
        raise ImportError("pip install redis") from exc

    url = vs_details.get("REDIS_URL") or vs_details.get("redis_url")

    if url:
        parsed = _parse_redis_url(url)
        host     = parsed["host"]
        port     = parsed["port"]
        password = parsed["password"]
        username = parsed["username"]
        db       = parsed["db"]
        ssl      = parsed["ssl"]
    else:
        host     = vs_details.get("REDIS_HOST") or vs_details.get("redis_host") or "localhost"
        port     = int(vs_details.get("REDIS_PORT") or vs_details.get("redis_port") or 6379)
        password = (
            vs_details.get("REDIS_PASSWORD")
            or vs_details.get("REDIS_PASS")
            or vs_details.get("redis_password")
            or vs_details.get("redis_pass")
        )
        db       = int(vs_details.get("REDIS_DB") or vs_details.get("redis_db") or 0)
        username = vs_details.get("REDIS_USERNAME") or vs_details.get("redis_username") or "default"
        ssl      = False

    return redis_lib.Redis(
        host=host, port=port, db=db,
        username=username, password=password,
        ssl=ssl,
        decode_responses=False,
        socket_connect_timeout=10,
        socket_timeout=30,
    )


def _redis_index_exists(r, index_name: str) -> bool:
    """Return True if the RediSearch index exists."""
    try:
        r.ft(index_name).info()
        return True
    except Exception:
        return False


def _redis_list_indexes(r) -> List[str]:
    """Return all RediSearch index names available."""
    try:
        raw = r.execute_command("FT._LIST")
        return [
            (n.decode("utf-8") if isinstance(n, bytes) else str(n))
            for n in (raw or [])
        ]
    except Exception:
        return []


def _redis_detect_vector_field(r, index_name: str) -> str:
    """
    Inspect an existing RediSearch index and return the name of the VECTOR
    field used for KNN search.

    Preference order:
      1. "embedding_dense"  — ingestion field name used by this system
      2. "embedding"        — legacy fallback
      3. first VECTOR field found in the schema

    Falls back to "embedding_dense" if the schema cannot be read.
    """
    PREFERRED = ("embedding_dense", "embedding")
    try:
        info = r.ft(index_name).info()
        # info is a dict (redis-py ≥ 4) or a flat list (older versions)
        if isinstance(info, dict):
            attributes = info.get("attributes") or info.get("fields") or []
        else:
            # flat list: find the "attributes" key and grab the following value
            try:
                idx = list(info).index(b"attributes")
                attributes = info[idx + 1]
            except (ValueError, IndexError):
                attributes = []

        vector_fields: List[str] = []
        for attr in attributes:
            # Each attribute is itself a flat list: [b'identifier', name, b'attribute', alias, b'type', type, ...]
            if isinstance(attr, (list, tuple)):
                parts = [
                    (p.decode("utf-8", errors="replace") if isinstance(p, bytes) else str(p))
                    for p in attr
                ]
                # Find pairs where the type value is VECTOR
                for i, part in enumerate(parts):
                    if part.upper() == "VECTOR" and i > 0:
                        # The field name is typically two positions before "type"
                        # Layout: identifier <name> attribute <alias> type VECTOR ...
                        # Grab both identifier and attribute values
                        for j in range(0, i - 1, 2):
                            if parts[j].upper() in ("IDENTIFIER", "ATTRIBUTE"):
                                field_name = parts[j + 1]
                                if field_name not in vector_fields:
                                    vector_fields.append(field_name)
                        break
            elif isinstance(attr, dict):
                if str(attr.get("type", "")).upper() == "VECTOR":
                    name = attr.get("identifier") or attr.get("attribute") or attr.get("name")
                    if name and name not in vector_fields:
                        vector_fields.append(str(name))

        for preferred in PREFERRED:
            if preferred in vector_fields:
                logger.info("Redis: KNN field detected as '%s' from index '%s'", preferred, index_name)
                return preferred

        if vector_fields:
            logger.info(
                "Redis: KNN field detected as '%s' (first VECTOR field) from index '%s'",
                vector_fields[0], index_name,
            )
            return vector_fields[0]

    except Exception as exc:
        logger.warning("Redis: could not inspect index schema for '%s': %s", index_name, exc)

    # Safe default — matches the ingestion field name used by this codebase
    return "embedding_dense"


def _redis_search(
    query_vector: List[float],
    vs_details: Dict[str, Any],
    collection: str,
    *,
    limit: int = 5,
    score_threshold: float = 0.0,
    is_active: Optional[bool] = True,
    filters: Optional[Dict[str, Any]] = None,
    version: Optional[int] = None,
    include_dense_values: bool = False,
    include_sparse_values: bool = False,
) -> List[RetrievalResult]:
    """
    Dense KNN search against a Redis / RediSearch index.

    Field layout expected in stored hashes (set by ingestion):
      embedding_dense   — FLOAT32 dense vector (used for KNN)
      embedding_sparse  — JSON string {"indices": [...], "values": [...]}
      sparse_values     — JSON list of floats (flat sparse weights, positional)
      text              — plain text chunk
      payload           — JSON metadata blob
      chunk_id          — unique chunk identifier

    The KNN vector field name is auto-detected from the existing index schema
    so that pre-built indexes (e.g. "485_215") work without any configuration.

    Sparse values are ALWAYS fetched from stored hashes so that sparse/hybrid
    callers (_redis_sparse_rescore + reciprocal_rank_fusion) have the data
    they need even when include_sparse_values=False was passed by the caller.
    """
    try:
        from redis.commands.search.query import Query
        from redis.commands.search.field import VectorField, TextField, TagField
        from redis.commands.search.index_definition import IndexDefinition, IndexType
    except ImportError as exc:
        raise ImportError("pip install redis") from exc

    # Canonical field names as stored during ingestion
    DENSE_VECTOR_FIELD   = "embedding_dense"    # FLOAT32 vector — KNN target
    SPARSE_VECTOR_FIELD  = "embedding_sparse"   # JSON {"indices":[…],"values":[…]}
    SPARSE_INDICES_FIELD = "sparse_indices"     # JSON list of ints  (alternative layout)
    SPARSE_VALUES_FIELD  = "sparse_values"      # JSON list of floats (alternative layout)

    r = _redis_connect(vs_details)

    # ── Resolve index name ──────────────────────────────────────────────────
    explicit_index = (
        vs_details.get("REDIS_INDEX_NAME") or vs_details.get("redis_index_name")
    )
    # collection itself is often the index name (e.g. "485_215")
    candidates = [c for c in [explicit_index, collection, f"idx:{collection}"] if c]

    index_name: Optional[str] = None
    for candidate in candidates:
        if _redis_index_exists(r, candidate):
            index_name = candidate
            logger.info(
                "Redis: using existing index '%s' for collection '%s'",
                index_name, collection,
            )
            break

    # ── Auto-create index if none found ────────────────────────────────────
    if index_name is None:
        index_name = f"idx:{collection}"
        logger.info(
            "Redis: no existing index found for collection '%s'. "
            "Creating '%s' with field '%s'.",
            collection, index_name, DENSE_VECTOR_FIELD,
        )
        import time as _time

        def _make_schema(algo: str) -> tuple:
            return (
                TagField("isActive", sortable=True),
                TextField("text", sortable=True, weight=1.0),
                VectorField(
                    DENSE_VECTOR_FIELD,
                    algo,
                    {
                        "TYPE": "FLOAT32",
                        "DIM": len(query_vector),
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )

        definition = IndexDefinition(
            prefix=[f"{collection}:"],
            index_type=IndexType.HASH,
        )
        created = False
        for algo in ("FLAT", "HNSW"):
            try:
                r.ft(index_name).create_index(
                    fields=_make_schema(algo), definition=definition
                )
                logger.info("✅ Redis: created index '%s' (algo=%s)", index_name, algo)
                _time.sleep(2)
                created = True
                break
            except Exception as exc:
                logger.warning(
                    "Redis: index creation with %s failed: %s", algo, exc
                )
        if not created:
            raise RuntimeError(
                f"Redis: could not create index '{index_name}' for collection '{collection}'. "
                "Check Redis connection and permissions."
            )

    # ── Detect the actual KNN vector field from the index schema ───────────
    # This handles pre-existing indexes that may use "embedding_dense",
    # "embedding", or any other name chosen during ingestion.
    knn_field = _redis_detect_vector_field(r, index_name)
    logger.info(
        "Redis: KNN will query field '%s' on index '%s'", knn_field, index_name
    )

    # ── Build filter string from isActive and user filters ───────────────────
    redis_filter = _build_redis_filter(filters, is_active, version)
    if redis_filter:
        filter_str = redis_filter
    else:
        filter_str = "*"

    # ── Vector bytes ────────────────────────────────────────────────────────
    from array import array as _array
    vec_bytes = _array("f", (float(v) for v in query_vector)).tobytes()

    # Fetch a wider candidate set so score_threshold / is_active filtering
    # still yields enough results after pruning.
    knn_limit = max(limit * 4, 20)

    # Always return sparse fields: sparse/hybrid modes rely on them for
    # _redis_sparse_rescore even when the caller did not set include_sparse_values.
    return_fields_list = [
        "chunk_id", "text", "metadata", "chunk_index", "vector_score",
        SPARSE_VECTOR_FIELD, SPARSE_INDICES_FIELD, SPARSE_VALUES_FIELD,
    ]
    if include_dense_values:
        return_fields_list.append(DENSE_VECTOR_FIELD)

    def _run_search(filt: str) -> Any:
        q_str = f"{filt}=>[KNN {knn_limit} @{knn_field} $query_vector AS vector_score]"
        q = (
            Query(q_str)
            .sort_by("vector_score")
            .return_fields(*return_fields_list)
            .paging(0, knn_limit)
            .dialect(2)
        )
        return r.ft(index_name).search(q, query_params={"query_vector": vec_bytes})

    try:
        raw = _run_search(filter_str)
    except Exception as exc:
        err = str(exc).lower()
        if is_active is not None and (
            "syntax error" in err or "wrong number" in err or "no results" in err
        ):
            logger.warning("Redis: isActive filter failed — retrying without filter")
            raw = _run_search("*")
        else:
            raise

    total_hits = getattr(raw, "total", len(getattr(raw, "docs", [])))
    logger.info(
        "Redis KNN | index=%s | knn_field=%s | filter=%s | raw_hits=%d | knn_limit=%d",
        index_name, knn_field, filter_str, total_hits, knn_limit,
    )

    def _decode(val, default="") -> str:
        if val is None:
            return default
        return val.decode("utf-8", errors="replace") if isinstance(val, bytes) else str(val)

    def _parse_json_field(raw_val: Any, default: Any) -> Any:
        decoded = _decode(raw_val, "")
        if not decoded:
            return default
        try:
            return json.loads(decoded)
        except json.JSONDecodeError:
            return default

    results: List[RetrievalResult] = []
    for doc in raw.docs:
        raw_score = float(getattr(doc, "vector_score", 2.0))
        # RediSearch COSINE distance: 0 = identical, 2 = opposite.
        # Convert to a [0, 1] similarity score.
        sim = max(0.0, 1.0 - (raw_score / 2.0))

        if sim < score_threshold:
            continue

        text = _decode(getattr(doc, "text", None))
        cid  = _decode(getattr(doc, "chunk_id", None))

        payload  = _parse_json_field(getattr(doc, "payload",  None), {})
        metadata = _parse_json_field(getattr(doc, "metadata", None), {})
        meta = payload if payload else metadata

        # ── Dense values (only when explicitly requested) ─────────────────
        dense_values = None
        if include_dense_values:
            dense_raw = getattr(doc, DENSE_VECTOR_FIELD, None)
            if dense_raw is not None:
                decoded_dense = _decode(dense_raw, "")
                if decoded_dense.startswith("["):
                    dense_values = _parse_json_field(dense_raw, [])
                else:
                    # Stored as raw bytes (FLOAT32 array) — convert back to list
                    try:
                        import struct
                        num_floats = len(dense_raw) // 4
                        dense_values = list(
                            struct.unpack(f"{num_floats}f", dense_raw)
                        )
                    except Exception:
                        dense_values = None
            if not isinstance(dense_values, list):
                dense_values = None

        # ── Sparse values — always parsed for re-scoring support ──────────
        sparse_values: Optional[Dict[str, List]] = None

        # Strategy 1: embedding_sparse is a JSON object {"indices":[],"values":[]}
        raw_sv = getattr(doc, SPARSE_VECTOR_FIELD, None)
        if raw_sv is not None:
            sparse_values = _parse_sparse_candidate(_decode(raw_sv, ""))

        # Strategy 2: separate sparse_indices / sparse_values fields
        if sparse_values is None:
            s_indices = _parse_json_field(getattr(doc, SPARSE_INDICES_FIELD, None), None)
            s_vals    = _parse_json_field(getattr(doc, SPARSE_VALUES_FIELD,  None), None)
            if isinstance(s_indices, list) and isinstance(s_vals, list):
                try:
                    sparse_values = _normalize_sparse_vector(s_indices, s_vals)
                except ValueError:
                    sparse_values = None

        results.append(RetrievalResult(
            score        = sim,
            text         = text,
            metadata     = meta,
            point_id     = cid or _decode(getattr(doc, "id", None)),
            dense_values = dense_values,
            # Expose sparse only when caller requested it; internally always
            # populated so _redis_sparse_rescore can read result.sparse_values.
            sparse_values = sparse_values,
        ))

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


# ========================= BACKEND ROUTER =========================

def _resolve_backend_id(vector_store_details: Dict[str, Any]) -> int:
    """
    Resolve which vector store backend to use.

      1 → Qdrant (internal)
      2 → Qdrant (user-provided)
      3 → Pinecone
      4 → Redis
      5 → PGVector
      6 → MongoDB Atlas
    """
    return int(vector_store_details.get("vector_store_id") or 1)


def _resolve_qdrant_url(
    vector_store_details: Dict[str, Any],
    backend_id: int,
    env: str = "uat",
) -> Tuple[str, Optional[str]]:
    """Return (base_url, api_key) for a Qdrant backend."""
    if backend_id == 2:
        url = (
            vector_store_details.get("QDRANT_URL")
            or vector_store_details.get("qdrant_url")
            or QDRANT_URLS.get(env, QDRANT_URLS["uat"])
        )
        api_key = (
            vector_store_details.get("QDRANT_API_KEY")
            or vector_store_details.get("qdrant_api_key")
        )
        return url.rstrip("/"), api_key
    # Internal Qdrant
    return _qdrant_base(env), None


def _mongo_tls_hint() -> str:
    return (
        "MongoDB Atlas TLS failed before authentication. "
        "Check the Atlas IP access list for this machine, outbound TLS inspection/proxy rules, "
        "and whether the deployment requires a client certificate. "
        "Optional retrieval_new.py vector_store_details keys supported here: "
        "MONGODB_TLS_CA_FILE, MONGODB_TLS_CERTIFICATE_KEY_FILE, "
        "MONGODB_TLS_CERTIFICATE_KEY_FILE_PASSWORD, "
        "MONGODB_TLS_ALLOW_INVALID_CERTIFICATES, MONGODB_TLS_ALLOW_INVALID_HOSTNAMES."
    )


def _is_mongo_tls_error(message: str) -> bool:
    """Return True when message looks like a TLS / certificate connectivity error."""
    lowered = str(message).lower()
    tls_keywords = (
        "tls", "ssl", "certificate", "handshake", "x509",
        "tlshandshake", "sslhandshakefailed", "tlsv1",
        "cert verify", "certificate verify", "self signed",
        "self-signed", "unable to get local issuer",
    )
    return any(kw in lowered for kw in tls_keywords)


# ========================= UNIFIED RETRIEVE =========================

def retrieve(
    query: str,
    collection: str,
    *,
    mode: str = "hybrid",
    dense_model: str = DEFAULT_DENSE_MODEL,
    env: str = "uat",
    limit: int = 5,
    score_threshold: float = 0.35,
    is_active: Optional[bool] = True,
    vector_store_details: Optional[Dict[str, Any]] = None,
    filters: Optional[Dict[str, Any]] = None,
    version: Optional[int] = None,
    point_ids: Optional[List[str]] = None,
    include_dense_values: bool = False,
    include_sparse_values: bool = False,
) -> List[RetrievalResult]:
    """
    Unified retrieval entry-point for all vector store backends.

    Parameters
    ----------
    query                : natural-language query string
    collection           : collection / table / namespace / index name
    mode                 : dense | sparse | hybrid | hybrid_rrf
    dense_model          : embedding model slug (must match ingestion, e.g. "bge-large")
    env                  : qdrant environment key ("dev" | "uat") or URL
    limit                : number of results to return
    score_threshold      : minimum similarity score (0–1)
    is_active            : filter on isActive flag (True/False/None for no filter)
    vector_store_details : dict from request_data["vector_store_details"]
    filters              : additional backend-specific filter dict
    point_ids            : optional point IDs to fetch directly (Pinecone only)
    include_dense_values : include stored dense values in results when supported
    include_sparse_values: include stored sparse values in results when supported
    """
    vs_details   = vector_store_details or {}
    backend_id   = _resolve_backend_id(vs_details)
    prefetch_lim = max(limit * 4, 20)   # wider net for RRF pre-fetch

    if point_ids and backend_id != 3:
        raise ValueError("Fetching stored point IDs is currently supported only for Pinecone")

    if point_ids and not include_dense_values and not include_sparse_values:
        include_sparse_values = True

    # Validate filters if provided
    if filters:
        backend_name = {
            1: "qdrant", 2: "qdrant", 3: "pinecone",
            4: "mongodb", 5: "pgvector", 6: "redis",
        }.get(backend_id, "all")
        filter_errors = _validate_filters(filters, backend_name)
        if filter_errors:
            raise ValueError(f"Invalid filter structure: {'; '.join(filter_errors)}")

    print(
        f"  [retrieval/{mode.upper()}] backend={backend_id} "
        f"collection={collection} ... ",
        end="", flush=True,
    )

    try:
        # ------------------------------------------------------------------
        # QDRANT INTERNAL  (backend_id 1)
        # Internal deployment — no API key, no auth headers.
        # Uses the legacy /points/search endpoint directly.
        # ------------------------------------------------------------------
        if backend_id == 1:
            # Internal Qdrant never requires an API key — use empty headers
            base_url = _qdrant_base(env)
            hdrs     = {}   # no authentication
            qf       = _apply_is_active_filter(filters, is_active, version)

            if mode == "dense":
                vec     = get_dense_embedding(query, dense_model)
                payload: dict = {
                    "vector":          {"name": DENSE_VECTOR_NAME, "vector": vec},
                    "limit":           limit,
                    "with_payload":    True,
                    "score_threshold": score_threshold,
                }
                if qf is not None:
                    payload["filter"] = qf
                results = _qdrant_search_with_search_api(base_url, collection, hdrs, payload)

            elif mode == "sparse":
                svec    = get_sparse_embedding(query)
                payload = {
                    "vector": {
                        "name":   SPARSE_VECTOR_NAME,
                        "vector": {"indices": svec["indices"], "values": svec["values"]},
                    },
                    "limit":        limit,
                    "with_payload": True,
                }
                if qf is not None:
                    payload["filter"] = qf
                results = _qdrant_search_with_search_api(base_url, collection, hdrs, payload)

            elif mode in {"hybrid", "hybrid_rrf"}:
                vec  = get_dense_embedding(query, dense_model)
                svec = get_sparse_embedding(query)
                dense_payload: dict = {
                    "vector":          {"name": DENSE_VECTOR_NAME, "vector": vec},
                    "limit":           prefetch_lim,
                    "with_payload":    True,
                    "score_threshold": score_threshold,
                }
                sparse_payload: dict = {
                    "vector": {
                        "name":   SPARSE_VECTOR_NAME,
                        "vector": {"indices": svec["indices"], "values": svec["values"]},
                    },
                    "limit":        prefetch_lim,
                    "with_payload": True,
                }
                if qf is not None:
                    dense_payload["filter"]  = qf
                    sparse_payload["filter"] = qf
                d_res   = _qdrant_search_with_search_api(base_url, collection, hdrs, dense_payload)
                s_res   = _qdrant_search_with_search_api(base_url, collection, hdrs, sparse_payload)
                results = reciprocal_rank_fusion([d_res, s_res], k=RRF_K)[:limit]

            else:
                raise ValueError(f"Unsupported mode '{mode}' for Qdrant internal backend")

        # ------------------------------------------------------------------
        # QDRANT USER-PROVIDED  (backend_id 2)
        # Uses the new /points/query API (Qdrant Cloud / v1.10+) with API key.
        # Vector names are auto-detected from the collection schema.
        # The isActive payload index is created proactively so the filter
        # always works on the first request without a round-trip failure.
        # ------------------------------------------------------------------
        elif backend_id == 2:
            base_url, api_key = _resolve_qdrant_url(vs_details, backend_id, env)

            if not api_key:
                logger.warning(
                    "Qdrant user-provided backend selected but QDRANT_API_KEY is "
                    "missing from vector_store_details. Requests may be rejected."
                )

            # Proactively ensure the isActive payload index exists so the
            # filter never causes a "Index required" 400 on first use.
            if is_active is not None:
                _qdrant_ensure_is_active_index(
                    base_url, collection, _qdrant_headers(api_key)
                )

            if mode == "dense":
                vec     = get_dense_embedding(query, dense_model)
                results = _qdrant_search_dense(
                    vec, collection, base_url,
                    limit=limit, score_threshold=score_threshold,
                    filters=filters, is_active=is_active, version=version, api_key=api_key,
                )

            elif mode == "sparse":
                svec    = get_sparse_embedding(query)
                results = _qdrant_search_sparse(
                    svec, collection, base_url,
                    limit=limit, filters=filters, is_active=is_active, api_key=api_key,
                )

            elif mode in {"hybrid", "hybrid_rrf"}:
                vec    = get_dense_embedding(query, dense_model)
                svec   = get_sparse_embedding(query)
                d_res  = _qdrant_search_dense(
                    vec, collection, base_url,
                    limit=prefetch_lim, score_threshold=score_threshold,
                    filters=filters, is_active=is_active, version=version, api_key=api_key,
                )
                s_res  = _qdrant_search_sparse(
                    svec, collection, base_url,
                    limit=prefetch_lim, filters=filters, is_active=is_active, api_key=api_key,
                )
                results = reciprocal_rank_fusion([d_res, s_res], k=RRF_K)[:limit]

            else:
                raise ValueError(f"Unsupported mode '{mode}' for Qdrant user-provided backend")

        # ------------------------------------------------------------------
        # PINECONE  (backend_id 3)
        # ------------------------------------------------------------------
        elif backend_id == 3:
            if point_ids:
                results = _pinecone_fetch_by_ids(
                    point_ids,
                    vs_details,
                    collection,
                    include_dense_values=include_dense_values,
                    include_sparse_values=include_sparse_values,
                )

            elif mode == "dense":
                vec     = get_dense_embedding(query, dense_model)
                results = _pinecone_search(
                    vec, vs_details, collection,
                    limit=limit, score_threshold=score_threshold, is_active=is_active,
                    filters=filters, version=version,
                )

            elif mode == "sparse":
                vec = get_dense_embedding(query, dense_model)
                svec = get_sparse_embedding(query)
                if _pinecone_index_supports_sparse_query(vs_details):
                    # Dense indexes cannot accept an empty vector, so keep the
                    # dense query alongside sparse values when native sparse is supported.
                    results = _pinecone_search(
                        vec, vs_details, collection,
                        limit=limit, score_threshold=score_threshold,
                        sparse_vec=svec, is_active=is_active,
                        filters=filters, version=version,
                    )
                else:
                    # Fallback for dense cosine indexes that store sparse values
                    # but cannot query them natively: dense candidate retrieval
                    # followed by local sparse re-scoring.
                    fallback_limit = max(limit * 10, 50)
                    logger.info(
                        "Pinecone sparse fallback | index does not support native "
                        "sparse query values, using dense prefetch + local sparse rescore"
                    )
                    d_res = _pinecone_search(
                        vec, vs_details, collection,
                        limit=fallback_limit, score_threshold=score_threshold,
                        is_active=is_active,
                        filters=filters, version=version,
                    )
                    results = _pinecone_sparse_rescore(d_res, svec, vs_details, collection)[:limit]

            elif mode in {"hybrid", "hybrid_rrf"}:
                vec = get_dense_embedding(query, dense_model)
                svec = get_sparse_embedding(query)
                if _pinecone_index_supports_sparse_query(vs_details):
                    # Pinecone hybrid: single query with both dense + sparse
                    results = _pinecone_search(
                        vec, vs_details, collection,
                        limit=limit, score_threshold=score_threshold,
                        sparse_vec=svec, is_active=is_active,
                        filters=filters, version=version,
                    )
                else:
                    logger.info(
                        "Pinecone hybrid fallback | index does not support native "
                        "sparse query values, using dense prefetch + local sparse RRF"
                    )
                    d_res = _pinecone_search(
                        vec, vs_details, collection,
                        limit=prefetch_lim, score_threshold=score_threshold,
                        is_active=is_active,
                        filters=filters, version=version,
                    )
                    s_res = _pinecone_sparse_rescore(d_res, svec, vs_details, collection)
                    results = reciprocal_rank_fusion([d_res, s_res], k=RRF_K)[:limit]

            else:
                raise ValueError(f"Unsupported mode '{mode}' for Pinecone backend")

            if (
                not point_ids
                and results
                and (include_dense_values or include_sparse_values)
            ):
                results = _enrich_pinecone_results_with_fetch(
                    results,
                    vs_details,
                    collection,
                    include_dense_values=include_dense_values,
                    include_sparse_values=include_sparse_values,
                )

        # ------------------------------------------------------------------
        # REDIS  (backend_id 6)
        # ------------------------------------------------------------------
        elif backend_id == 6:
            vec = get_dense_embedding(query, dense_model)

            if mode == "dense":
                results = _redis_search(
                    vec, vs_details, collection,
                    limit=limit,
                    score_threshold=score_threshold,
                    is_active=is_active,
                    filters=filters, version=version,
                    include_dense_values=include_dense_values,
                    include_sparse_values=include_sparse_values,
                )

            elif mode == "sparse":
                logger.info(
                    "Redis sparse fallback | using dense prefetch + stored sparse rescore"
                )
                fallback_limit = max(limit * 10, 50)
                d_res = _redis_search(
                    vec, vs_details, collection,
                    limit=fallback_limit,
                    score_threshold=score_threshold,
                    is_active=is_active,
                    filters=filters, version=version,
                    include_dense_values=include_dense_values,
                    include_sparse_values=True,
                )
                svec = get_sparse_embedding(query)
                results = _redis_sparse_rescore(d_res, svec)[:limit]

            elif mode in {"hybrid", "hybrid_rrf"}:
                logger.info(
                    "Redis hybrid fallback | using dense prefetch + stored sparse RRF"
                )
                d_res = _redis_search(
                    vec, vs_details, collection,
                    limit=prefetch_lim,
                    score_threshold=score_threshold,
                    is_active=is_active,
                    filters=filters, version=version,
                    include_dense_values=include_dense_values,
                    include_sparse_values=True,
                )
                svec = get_sparse_embedding(query)
                s_res = _redis_sparse_rescore(d_res, svec)
                results = reciprocal_rank_fusion([d_res, s_res], k=RRF_K)[:limit]

            else:
                raise ValueError(f"Unsupported mode '{mode}' for Redis backend")

        # ------------------------------------------------------------------
        # PGVECTOR  (backend_id 5)
        # ------------------------------------------------------------------
        elif backend_id == 5:
            vec = get_dense_embedding(query, dense_model)

            if mode == "sparse":
                # PGVector has no native sparse ANN index.
                # Strategy: dense prefetch over a wide candidate pool, then
                # dot-product re-score using stored sparse_indices/sparse_values,
                # then return top-k by sparse score — mirrors Redis sparse mode.
                logger.warning(
                    "PGVector has no native sparse index; "
                    "using dense prefetch + stored sparse dot-product rescore."
                )
                fallback_limit = max(limit * 10, 50)
                d_res = _pgvector_search(
                    vec, vs_details, collection,
                    limit=fallback_limit, score_threshold=score_threshold, is_active=is_active,
                    filters=filters, version=version,
                )
                svec = get_sparse_embedding(query)
                results = _pgvector_sparse_rescore(d_res, svec, vs_details, collection)[:limit]

            elif mode in {"hybrid", "hybrid_rrf"}:
                # PGVector stores sparse_indices/sparse_values but has no native
                # sparse ANN index. Run dense search + in-memory RRF using stored
                # sparse scores as the second list.
                d_res = _pgvector_search(
                    vec, vs_details, collection,
                    limit=prefetch_lim, score_threshold=score_threshold, is_active=is_active,
                    filters=filters, version=version,
                )
                # Re-score with stored sparse vectors for RRF second list
                svec   = get_sparse_embedding(query)
                s_res  = _pgvector_sparse_rescore(
                    d_res, svec, vs_details, collection,
                )
                results = reciprocal_rank_fusion([d_res, s_res], k=RRF_K)[:limit]
            else:
                results = _pgvector_search(
                    vec, vs_details, collection,
                    limit=limit, score_threshold=score_threshold, is_active=is_active,
                    filters=filters, version=version,
                )

        # ------------------------------------------------------------------
        # MONGODB ATLAS  (backend_id 4)
        # MongoDB has no native sparse ANN index. Sparse mode is emulated
        # via dense prefetch + stored sparse dot-product re-scoring (same
        # pattern as Redis/PGVector). Hybrid uses RRF over both lists.
        # ------------------------------------------------------------------
        elif backend_id == 4:
            vec = get_dense_embedding(query, dense_model)

            if mode == "dense":
                results = _mongo_search(
                    vec, vs_details, collection,
                    limit=limit, score_threshold=score_threshold, is_active=is_active,
                    filters=filters, version=version,
                )

            elif mode == "sparse":
                # MongoDB has no native sparse ANN index.
                # Strategy: dense prefetch over a wide candidate pool, then dot-product
                # re-score using stored sparse vectors, then return top-k by sparse score.
                # MONGODB_DENSE_FIELD must point to the actual dense float-array field
                # (default "dense"). The guard inside _mongo_search auto-corrects the
                # common misconfiguration where MONGODB_DENSE_FIELD="sparse".
                logger.info(
                    "MongoDB sparse mode | dense prefetch + stored sparse dot-product rescore"
                )
                fallback_limit = max(limit * 10, 50)
                svec = get_sparse_embedding(query)
                d_res = _mongo_search(
                    vec, vs_details, collection,
                    limit=fallback_limit, score_threshold=0.0, is_active=is_active,
                    filters=filters, version=version,
                )
                rescored = _mongo_sparse_rescore(d_res, svec, vs_details, collection)
                # Apply score_threshold on the final sparse scores
                results = [r for r in rescored if r.score >= score_threshold][:limit]

            elif mode in {"hybrid", "hybrid_rrf"}:
                d_res  = _mongo_search(
                    vec, vs_details, collection,
                    limit=prefetch_lim, score_threshold=score_threshold, is_active=is_active,
                    filters=filters, version=version,
                )
                svec   = get_sparse_embedding(query)
                s_res  = _mongo_sparse_rescore(d_res, svec, vs_details, collection)
                results = reciprocal_rank_fusion([d_res, s_res], k=RRF_K)[:limit]

            else:
                raise ValueError(f"Unsupported mode '{mode}' for MongoDB Atlas backend")

        else:
            raise ValueError(f"Unknown vector_store_id: {backend_id}")

        print(f"found {len(results)} chunk(s)")
        return results

    except Exception as exc:
        message = str(exc)
        if backend_id == 4 and _is_mongo_tls_error(message):
            hint = _mongo_tls_hint()
            print("ERROR: MongoDB Atlas TLS handshake failed.")
            print(f"HINT: {hint}")
            logger.error("Retrieval failed: MongoDB Atlas TLS handshake failed. %s", hint)
            raise RetrievalExecutionError(
                f"MongoDB Atlas TLS handshake failed. {hint}"
            ) from exc
        else:
            print(f"ERROR: {message}")
            logger.error("Retrieval failed: %s", exc, exc_info=True)
            raise RetrievalExecutionError(message) from exc


# ========================= SPARSE RE-SCORE HELPERS =========================
# For backends without native sparse ANN (PGVector, MongoDB Atlas) we fetch
# stored sparse vectors, compute dot-product similarity against the query
# sparse vector, and feed the scored results into RRF as the "sparse list".

def _dot_product(
    query_indices: List[int],
    query_values: List[float],
    doc_indices: List[int],
    doc_values: List[float],
) -> float:
    """Compute dot product between two sparse vectors."""
    q_map = dict(zip(query_indices, query_values))
    return sum(q_map.get(i, 0.0) * v for i, v in zip(doc_indices, doc_values))


def _pgvector_sparse_rescore(
    dense_results: List[RetrievalResult],
    query_sparse: Dict[str, List],
    vs_details: Dict[str, Any],
    collection: str,
) -> List[RetrievalResult]:
    """
    Fetch stored sparse vectors for the dense-result chunk_ids from PGVector
    and return a re-scored list for RRF fusion.
    """
    if not dense_results:
        return []

    try:
        import psycopg2
        import psycopg2.extras
    except ImportError:
        return []

    import re
    table = re.sub(r"[^a-zA-Z0-9_]", "_", collection)

    chunk_ids = [r.point_id for r in dense_results if r.point_id]
    if not chunk_ids:
        return []

    conn = _pgvector_connect(vs_details)

    placeholders = ", ".join(["%s"] * len(chunk_ids))
    sql = f"""
        SELECT chunk_id, sparse_indices, sparse_values
        FROM "{table.replace(chr(34), chr(34) * 2)}"
        WHERE chunk_id IN ({placeholders});
    """

    q_indices = query_sparse.get("indices", [])
    q_values  = query_sparse.get("values", [])
    scored_map: Dict[str, RetrievalResult] = {r.point_id: r for r in dense_results if r.point_id}
    results: List[RetrievalResult] = []

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, chunk_ids)
            for row in cur.fetchall():
                cid  = str(row["chunk_id"])   # cast UUID → str to match scored_map keys
                orig = scored_map.get(cid)
                if orig is None:
                    continue
                doc_indices = row.get("sparse_indices") or []
                doc_values  = row.get("sparse_values") or []
                sparse_score = _dot_product(q_indices, q_values, doc_indices, doc_values)
                results.append(RetrievalResult(
                    score    = sparse_score,
                    text     = orig.text,
                    metadata = orig.metadata,
                    point_id = cid,
                ))
    finally:
        conn.close()

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _mongo_sparse_rescore(
    dense_results: List[RetrievalResult],
    query_sparse: Dict[str, List],
    vs_details: Dict[str, Any],
    collection: str,
) -> List[RetrievalResult]:
    """
    Fetch stored sparse vectors from MongoDB and return a re-scored list for RRF.
    """
    if not dense_results:
        return []

    uri      = vs_details.get("MONGODB_URI") or vs_details.get("mongodb_uri")
    database = vs_details.get("MONGODB_DATABASE") or vs_details.get("mongodb_database")
    sparse_field = vs_details.get("MONGODB_SPARSE_FIELD") or vs_details.get("mongodb_sparse_field") or "sparse"

    chunk_ids = [r.point_id for r in dense_results if r.point_id]
    if not chunk_ids:
        return []

    try:
        client = _create_mongo_atlas_client(uri, vs_details=vs_details)
    except ImportError:
        return []

    col     = client[database][collection]
    docs    = list(col.find(
        {"chunk_id": {"$in": chunk_ids}},
        {"chunk_id": 1, sparse_field: 1, "text": 1},
    ))
    client.close()

    q_indices  = query_sparse.get("indices", [])
    q_values   = query_sparse.get("values", [])
    scored_map = {r.point_id: r for r in dense_results if r.point_id}
    results: List[RetrievalResult] = []

    for doc in docs:
        cid  = str(doc.get("chunk_id", ""))
        orig = scored_map.get(cid)
        if orig is None:
            continue
        sv = doc.get(sparse_field) or {}
        doc_indices = sv.get("indices", [])
        doc_values  = sv.get("values", [])
        sparse_score = _dot_product(q_indices, q_values, doc_indices, doc_values)
        results.append(RetrievalResult(
            score    = sparse_score,
            text     = orig.text,
            metadata = orig.metadata,
            point_id = cid,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _pinecone_sparse_rescore(
    dense_results: List[RetrievalResult],
    query_sparse: Dict[str, List],
    vs_details: Dict[str, Any],
    collection: str,
) -> List[RetrievalResult]:
    """
    Fetch stored sparse vectors from Pinecone and return a re-scored list.

    This is used as a fallback when the Pinecone index stores sparse values
    but does not support native sparse queries, such as dense cosine indexes.
    """
    if not dense_results:
        return []

    fetched_results = _pinecone_fetch_by_ids(
        [r.point_id for r in dense_results if r.point_id],
        vs_details,
        collection,
        include_sparse_values=True,
    )
    fetched_map = {r.point_id: r for r in fetched_results if r.point_id}

    q_indices = query_sparse.get("indices", [])
    q_values = query_sparse.get("values", [])
    results: List[RetrievalResult] = []

    for dense_result in dense_results:
        if not dense_result.point_id:
            continue
        fetched = fetched_map.get(dense_result.point_id)
        if fetched is None:
            continue
        sparse_payload = fetched.sparse_values or {}
        doc_indices = sparse_payload.get("indices", [])
        doc_values = sparse_payload.get("values", [])
        sparse_score = _dot_product(q_indices, q_values, doc_indices, doc_values)
        results.append(RetrievalResult(
            score=sparse_score,
            text=dense_result.text,
            metadata=dense_result.metadata,
            point_id=dense_result.point_id,
            sparse_values=fetched.sparse_values,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


def _redis_sparse_rescore(
    dense_results: List[RetrievalResult],
    query_sparse: Dict[str, List],
) -> List[RetrievalResult]:
    """Re-score Redis dense candidates using stored sparse vectors."""
    if not dense_results:
        return []

    q_indices = query_sparse.get("indices", [])
    q_values = query_sparse.get("values", [])
    results: List[RetrievalResult] = []

    for dense_result in dense_results:
        sparse_payload = dense_result.sparse_values or {}
        doc_indices = sparse_payload.get("indices", [])
        doc_values = sparse_payload.get("values", [])
        sparse_score = _dot_product(q_indices, q_values, doc_indices, doc_values)
        results.append(RetrievalResult(
            score=sparse_score,
            text=dense_result.text,
            metadata=dense_result.metadata,
            point_id=dense_result.point_id,
            dense_values=dense_result.dense_values,
            sparse_values=dense_result.sparse_values,
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    return results


# ========================= ENTITY (PLAYER) RE-RANK =========================
# Dense/sparse retrieval can rank one player's stat table above another's when
# the tables are near-identical in embedding space. When the query names a
# known entity, we re-rank so chunks that actually mention that entity come
# first (hard filter when any match, soft boost otherwise). This is backend-
# agnostic and does not depend on a dedicated metadata field — it matches the
# entity name already present in each chunk's text/heading (see pdf_extractor
# which prepends "<Player> — <Section>" to every table).

# Known entities and their aliases/nicknames. Extendable; matching is
# case-insensitive and alias-aware. Keyed by the canonical full name.
_ENTITY_ALIASES: Dict[str, List[str]] = {
    "Rohit Sharma": ["rohit sharma", "rohit", "hitman"],
    "Virat Kohli": ["virat kohli", "virat", "kohli"],
    "MS Dhoni": ["ms dhoni", "mahendra singh dhoni", "dhoni", "mahi"],
    "Hardik Pandya": ["hardik pandya", "hardik", "pandya"],
    "Ravindra Jadeja": ["ravindra jadeja", "jadeja", "jaddu", "sir jadeja"],
}


def _detect_entities(query: str) -> List[str]:
    """Return canonical entity names referenced in the query (longest-first).

    Uses word-boundary matching so 'jadeja' matches but substrings inside other
    words do not. Multiple entities can be returned (e.g. a comparison query).
    """
    if not query:
        return []
    import re as _re

    low = query.lower()
    found: List[str] = []
    for canonical, aliases in _ENTITY_ALIASES.items():
        for alias in aliases:
            if _re.search(rf"\b{_re.escape(alias)}\b", low):
                found.append(canonical)
                break
    return found


def _chunk_mentions_entity(text: str, canonical: str) -> bool:
    """True when a chunk's text mentions the entity (by any alias)."""
    if not text:
        return False
    import re as _re

    low = text.lower()
    for alias in _ENTITY_ALIASES.get(canonical, [canonical.lower()]):
        if _re.search(rf"\b{_re.escape(alias)}\b", low):
            return True
    return False


# Exact-identifier detection. Dense/semantic search cannot reliably retrieve a
# specific record by its ID (e.g. "EMP0001") because every record embeds almost
# identically — the ID is a token, not a concept. When a query contains an
# ID-like code, we hard-filter results to chunks that contain that exact ID.
# Pattern: letter-prefixed codes (EMP0001, INV2024), code-with-separators
# (INV-2024-01, PO_1234), or standalone long digit runs (order #100245).
_IDENTIFIER_RE = __import__("re").compile(
    r"\b("
    r"[A-Za-z]{2,}[-_/]?\d{2,}(?:[-_/]\w+)*"   # EMP0001, INV-2024-01, PO_1234
    r"|\d{5,}"                                   # bare long numeric ids
    r")\b"
)
# Common words that match the letter+digit shape but are not record IDs.
_IDENTIFIER_STOPWORDS = {"covid19", "mp3", "mp4", "h2o", "co2", "g20", "t20", "t20i", "u19"}


def _detect_identifiers(query: str) -> List[str]:
    """Return exact ID-like tokens in the query (e.g. 'EMP0001')."""
    if not query:
        return []
    found: List[str] = []
    for m in _IDENTIFIER_RE.finditer(query):
        tok = m.group(1)
        if tok.lower() in _IDENTIFIER_STOPWORDS:
            continue
        if tok not in found:
            found.append(tok)
    return found


def _chunk_contains_identifier(text: str, ident: str) -> bool:
    """True when a chunk's text contains the exact identifier (case-insensitive)."""
    if not text:
        return False
    import re as _re
    return _re.search(rf"\b{_re.escape(ident)}\b", text, _re.IGNORECASE) is not None


def _rerank_by_entity(query: str, results: List["RetrievalResult"]) -> List["RetrievalResult"]:
    """Re-rank results so query-named entities / exact IDs come first.

    Priority:
      1. Exact identifiers (EMP0001, invoice #, ...) — hard-filter to chunks
         that contain the ID. This fixes record-lookup queries that dense
         search cannot resolve. If none contain it, fall back to original order.
      2. Named entities (players, etc.) — same hard-filter behaviour.
      3. Otherwise, unchanged.
    """
    if not results:
        return results

    # --- 1. Exact identifier lookup takes precedence ----------------------
    identifiers = _detect_identifiers(query)
    if identifiers:
        def contains_any_id(r) -> bool:
            md = r.metadata or {}
            blob = (r.text or "") + " " + " ".join(
                str(md.get(k, "")) for k in ("heading", "section", "player", "entity")
            )
            return any(_chunk_contains_identifier(blob, i) for i in identifiers)

        id_matches = [r for r in results if contains_any_id(r)]
        if id_matches:
            return id_matches  # only the exact record(s) requested

    # --- 2. Named-entity re-rank ------------------------------------------
    entities = _detect_entities(query)
    if not entities:
        return results

    def mentions_any(r) -> bool:
        text = (r.text or "")
        md = r.metadata or {}
        extra = " ".join(
            str(md.get(k, "")) for k in ("heading", "section", "player", "entity")
        )
        blob = f"{text} {extra}"
        return any(_chunk_mentions_entity(blob, e) for e in entities)

    matching = [r for r in results if mentions_any(r)]
    non_matching = [r for r in results if not mentions_any(r)]

    if not matching:
        return results

    if len(entities) == 1:
        return matching

    return matching + non_matching


# ========================= TEXT FORMATTING =========================

def _to_plain_text(text: str) -> str:
    """Flatten Markdown-ish chunk text into a single plain-text paragraph.

    Removes structural markup so the result reads as continuous prose:
      - heading markers (#, ##, ...) and bold/italic emphasis (**, *, _, `)
      - known label prefixes (e.g. "Caption/Label >")
      - Markdown table pipes and the |---|---| separator rows, keeping the
        cell text joined by spaces
      - bullet / blockquote markers (-, *, >) at line starts
      - collapses all whitespace/newlines into single spaces

    Intended for display only; the stored chunk text is left unchanged.
    """
    import re

    if not text:
        return ""

    out_lines: List[str] = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue

        # Drop a Markdown table separator row like |---|:--:|
        if re.fullmatch(r"\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?", line):
            continue

        # Markdown table row: split on pipes, keep non-empty cells.
        if line.count("|") >= 2:
            cells = [c.strip() for c in line.strip("|").split("|")]
            line = " ".join(c for c in cells if c)

        # Strip leading heading / bullet / blockquote markers.
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^[>\-\*\u2022]+\s*", "", line)

        # Drop a leading "Caption/Label >" style prefix.
        line = re.sub(r"^[A-Za-z][\w /]*\s*>\s*", "", line)

        out_lines.append(line)

    joined = " ".join(out_lines)

    # Remove inline emphasis / code markers.
    joined = re.sub(r"[*_`]+", "", joined)
    # Collapse any remaining whitespace runs.
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined


# ========================= PROMPT BUILDING =========================

def build_augmented_user_message(question: str, chunks: List[RetrievalResult]) -> str:
    if not chunks:
        return (
            "No relevant information was found in the knowledge base.\n\n"
            f"Question: {question}"
        )
    ctx_lines = [
        f"[{i}] (score {c.score:.4f}) {c.text.strip()}"
        for i, c in enumerate(chunks, 1)
    ]
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not present in the context, say "
        "'I could not find any information about this in the knowledge base.'\n\n"
        f"--- CONTEXT ---\n{''.join(ctx_lines)}\n"
        "--- END CONTEXT ---\n\n"
        f"Question: {question}"
    )


def format_llama3_chat(messages: list) -> str:
    out = []
    i   = 0
    if messages and messages[0]["role"] == "system":
        out.append(f"{BOS}{SOH}system{EOH}\n\n{messages[0]['content']}{EOT}")
        i = 1
    else:
        out.append(BOS)
    while i < len(messages):
        m = messages[i]
        if m["role"] == "user":
            out.append(f"{SOH}user{EOH}\n\n{m['content']}{EOT}")
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                out.append(f"{SOH}{ASSISTANT_ROLE}{EOH}\n\n{messages[i+1]['content']}{EOT}")
                i += 2
            else:
                out.append(f"{SOH}{ASSISTANT_ROLE}{EOH}\n\n")
                i += 1
        else:
            i += 1
    return "".join(out)


def truncate_history(messages: list, max_pairs: int = 8) -> list:
    sys_msg = [m for m in messages if m["role"] == "system"][:1]
    conv    = [m for m in messages if m["role"] != "system"]
    return sys_msg + conv[-(max_pairs * 2):]


# ========================= FOLLOW-UP QUESTIONS =========================
# A follow-up such as "how old is he?" or "and the second one?" cannot be
# searched as written: the vector store has no idea who "he" is. The caller
# sends the earlier turns as `history`. We rewrite the new message into a
# standalone query for the SEARCH, and still hand the answering model the real
# conversation so it can resolve references itself.

_HISTORY_ROLES = {"user", "assistant"}
# A "message" is one user OR one assistant turn, so 20 messages = 10 exchanges.
# Too small a window silently mis-resolves references to older turns ("the first
# one you mentioned" lands on whoever is the earliest turn still visible).
_REWRITE_MAX_TURNS = int(os.getenv("REWRITE_HISTORY_MESSAGES", "20"))   # shown to the rewriter
_REWRITE_TURN_CHARS = 600         # per message (assistant answers can be long)
_REWRITE_MAX_QUERY_CHARS = 500
_ANSWER_HISTORY_MESSAGES = int(os.getenv("ANSWER_HISTORY_MESSAGES", "16"))  # sent to the answering model
_ANSWER_TURN_CHARS = 2000

# Returned by the rewriter when the latest message only asks to reshape the
# previous answer (any format, structure, length or tone). Nothing new
# needs to be looked up, so retrieval and external APIs are skipped.
NO_SEARCH = "NO_SEARCH"

_RESHAPE_SYSTEM_PROMPT = (
    "You are a helpful assistant. The user is asking you to reshape or restate your "
    "earlier answer in the conversation: a different format, structure, "
    "length or tone (for example a table, bullet list, JSON, YAML, CSV, XML, "
    "markdown, an email or a summary). Use ONLY the facts already "
    "in the conversation: do not add, drop or change facts, and do not look "
    "anything up. Reply in the requested form and nothing else. The requested "
    "form applies to this reply only. If the requested form needs information "
    "that is not in the earlier answer, or the earlier answer says nothing was "
    "found or reports an error, say so plainly instead of guessing."
)

_REWRITE_SYSTEM_PROMPT = (
    "You rewrite the user's latest message into ONE standalone search query for a "
    "knowledge base.\n"
    "- Resolve references to earlier turns: pronouns (he, she, it, they, this, that), "
    "phrases like 'the first one', 'the same for X', 'what about ...', 'and in 2023?'.\n"
    "- Pronouns (it, its, this, that, them) refer to the MOST RECENT exchange unless "
    "the message clearly points further back.\n"
    "- Keep every specific name, ID, number, date and technical term from the latest "
    "message, plus the ones from earlier turns that it refers to.\n"
    "- Drop instructions about the answer's presentation (format, structure, "
    "length, tone, e.g. 'as a table', 'in JSON', 'shorter') from the query: they say "
    "how to answer, not what to search for.\n"
    "- If the latest message ONLY asks to reformat, restate, summarize, shorten, "
    "expand or continue the assistant's previous answer, AND every fact "
    f"it needs is already in that answer, output exactly {NO_SEARCH}. If it asks "
    "for anything not in that answer (an extra detail, another item, a comparison, "
    "fresher data), or the previous answer found nothing or failed, write a normal "
    "search query instead.\n"
    "- If the latest message is already self-contained, return it unchanged.\n"
    "- Do NOT answer the question and do NOT add facts that were not in the conversation.\n"
    "- Output only the rewritten query on a single line: no quotes, no explanation."
)


def _clip_turn(text: str, limit: int) -> str:
    """Shorten a long turn but keep both ends: the conclusion of an answer
    matters as much as its opening."""
    if len(text) <= limit:
        return text
    marker = " ... "
    head = (limit - len(marker)) * 2 // 3
    tail = limit - len(marker) - head
    return text[:head] + marker + text[len(text) - tail:]


def _clean_history(history: Any) -> List[Dict[str, str]]:
    """Normalise client-supplied history to [{'role','content'}, ...].

    Accepts dicts or objects with role/content, keeps only user/assistant turns
    with text, and drops leading assistant turns (the model API requires the
    first message to come from the user).
    """
    cleaned: List[Dict[str, str]] = []
    for turn in history or []:
        if isinstance(turn, dict):
            role, content = turn.get("role", ""), turn.get("content", "")
        else:
            role, content = getattr(turn, "role", ""), getattr(turn, "content", "")
        role = str(role).strip().lower()
        content = str(content or "").strip()
        if role not in _HISTORY_ROLES or not content:
            continue
        if not cleaned and role != "user":
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def rewrite_followup_query(
    question: str,
    history: Any,
    generate_fn: Optional[Any] = None,
) -> str:
    """Return a standalone version of ``question`` using the earlier ``history``.

    With no history the question is returned unchanged. Never raises: if the
    model call fails or returns something unusable (empty, or far longer than the
    question, which usually means it answered instead of rewriting), the original
    question is searched as written.
    """
    q = (question or "").strip()
    turns = _clean_history(history)
    if not q or not turns:
        return q

    generate = generate_fn or _generate_answer_gpt
    recent = turns[-_REWRITE_MAX_TURNS:]
    conversation = "\n".join(
        f"{t['role'].capitalize()}: {_clip_turn(t['content'], _REWRITE_TURN_CHARS)}" for t in recent
    )
    messages = [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Conversation so far:\n{conversation}\n\n"
            f"Latest message: {q}\n\nStandalone query:"
        )},
    ]
    try:
        raw = generate(messages)
    except Exception as exc:
        logger.warning("Follow-up rewrite failed, searching the question as written: %s", exc)
        return q

    import re as _re
    line = next((ln.strip() for ln in str(raw or "").splitlines() if ln.strip()), "")
    line = _re.sub(r"^(standalone query|rewritten query|query)\s*:\s*", "", line, flags=_re.IGNORECASE)
    line = line.strip().strip("\"'`").strip()
    if line.strip(" .`").upper() == NO_SEARCH:
        # Only meaningful when there is an earlier answer to reshape.
        return NO_SEARCH if any(t["role"] == "assistant" for t in turns) else q
    if not line or len(line) > max(_REWRITE_MAX_QUERY_CHARS, len(q) * 6):
        return q
    return line


def _history_messages(history: Any) -> List[Dict[str, str]]:
    """Recent history as chat messages for the answering model (long turns clipped)."""
    turns = _clean_history(history)[-_ANSWER_HISTORY_MESSAGES:]
    while turns and turns[0]["role"] != "user":   # trimming may expose a leading assistant turn
        turns = turns[1:]
    return [{"role": t["role"], "content": _clip_turn(t["content"], _ANSWER_TURN_CHARS)} for t in turns]


# The answering model copies the look of earlier answers (a table answer is
# followed by another table). For a normal question we therefore show it the
# earlier turns as plain text, so a format asked for once does not stick.
_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")
_FORMAT_PHRASE_RE = re.compile(
    r"[\s,]*\b(?:and\s+)?(?:as|in|into|using)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:table|tabular\s+form|json|yaml|csv|xml|markdown|bullet(?:ed)?(?:\s+points?|\s+list)?"
    r"|numbered\s+list|list)(?:\s+(?:format|form))?\b",
    re.IGNORECASE,
)


def _plain_text(text: str) -> str:
    """Strip markdown/code-fence/table/bullet markup, keeping the words."""
    lines = []
    for line in text.splitlines():
        st = line.strip()
        if st.startswith("```") or re.fullmatch(r"[-*_]{3,}", st):
            continue
        if st.startswith("|") and _TABLE_SEPARATOR_RE.match(st):
            continue
        if st.startswith("|") and st.endswith("|"):
            cells = [c.strip() for c in st.strip("|").split("|") if c.strip()]
            line = ": ".join(cells) if len(cells) == 2 else ", ".join(cells)
        line = re.sub(r"\*\*|__|`", "", line)
        line = re.sub(r"^\s*(#{1,6}\s+|[-*]\s+)", "", line)
        lines.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _deformat_history(conversation: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Earlier turns without presentation: plain assistant text, and user turns
    without 'as a table' / 'in JSON' style phrases."""
    out = []
    for t in conversation:
        if t["role"] == "assistant":
            content = _plain_text(t["content"])
        else:
            content = _FORMAT_PHRASE_RE.sub("", t["content"]).strip()
        out.append({"role": t["role"], "content": content or t["content"]})
    return out


_ANSWER_STYLE_NOTE = (
    " Answer directly: do not mention the context, the tool result or that a call "
    "was made. Use a particular format (table, JSON, bullets, ...) only "
    "if the user's LATEST message asks for it; earlier turns' formats do not carry "
    "over."
)

_HISTORY_SYSTEM_NOTE = (
    " The earlier conversation is provided only so you can understand what the "
    "user's latest question refers to. State facts from the provided context; if "
    "the question is only about something said earlier in the conversation, you may "
    "use the conversation."
)


# ========================= LLM ANSWER GENERATION =========================

def _public_error(exc: Exception) -> str:
    """Error text that is safe to return to a caller: no internal URLs."""
    return re.sub(r"https?://\S+", "<url>", str(exc))


def _generate_answer_gpt(
    messages: List[Dict[str, str]],
    *,
    api_key: Optional[str] = None,      # unused for Bedrock; kept for call-site compatibility
    base_url: Optional[str] = None,     # unused for Bedrock
    model: Optional[str] = None,        # optional Bedrock model-id override
) -> str:
    """
    Generate a final answer using AWS Bedrock (Anthropic Claude).

    Bedrock is always used (no OpenAI). Credentials come from the standard AWS
    chain (env / profile / role) that boto3 resolves. `model` optionally
    overrides the Bedrock model id for this call. The legacy api_key/base_url
    params are accepted but ignored so existing call sites keep working.
    """
    try:
        import boto3
    except ImportError:
        logger.error("boto3 not installed — cannot generate answer via Bedrock")
        return "Answer generation unavailable (boto3 not installed)."

    model_id = (model or BEDROCK_CHAT_MODEL)

    # Anthropic on Bedrock wants the system prompt separate from the turn list.
    system_parts: List[str] = []
    anthropic_messages: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role in ("user", "assistant"):
            anthropic_messages.append({
                "role": role,
                "content": [{"type": "text", "text": content}],
            })
    if not anthropic_messages:
        anthropic_messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]

    request_body: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": BEDROCK_CHAT_MAX_TOKENS,
        "messages": anthropic_messages,
        "temperature": 0.2,   # Sonnet 4.x rejects temperature + top_p together — send only one
    }
    if system_parts:
        request_body["system"] = "\n\n".join(system_parts)

    try:
        client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
        response = client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
            contentType="application/json",
            accept="application/json",
        )
        result = json.loads(response["body"].read())

        # Best-effort token tracking (never fatal).
        try:
            try:
                from retrieval._vendor.token_tracker import get_token_tracker
            except ImportError:
                try:
                    from _vendor.token_tracker import get_token_tracker  # type: ignore
                except ImportError:
                    from services.token_tracker import get_token_tracker  # type: ignore
            get_token_tracker().add_from_bedrock_usage(result, source="answer_generation")
        except Exception:
            pass

        parts = [
            block.get("text", "")
            for block in result.get("content", [])
            if block.get("type") == "text"
        ]
        answer = "\n".join(parts).strip()
        return answer or "I could not generate an answer."

    except Exception as exc:
        logger.error("Bedrock answer generation failed: %s", exc, exc_info=True)
        return "Answer generation failed due to an internal error."


# ========================= LLM STREAMING (legacy) =========================

def stream_from_triton(base_url: str, model_name: str, prompt: str, sampling: dict):
    url     = f"{base_url}/v2/models/{model_name}/generate_stream"
    payload = {
        "text_input":          prompt,
        "parameters":          {"stream": True},
        "sampling_parameters": json.dumps(sampling),
    }
    with requests.post(url, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if line.startswith("data:"):
                try:
                    yield json.loads(line[5:].strip())
                except json.JSONDecodeError:
                    pass


# ========================= MAIN =========================

def main():
    parser = argparse.ArgumentParser(description="Multi-backend RAG Chat with Hybrid Search")
    parser.add_argument("--collection",  default="kb_2012",
                        help="Collection / table / namespace name")
    parser.add_argument("--embed-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--env",         default="uat", choices=["dev", "uat"],
                        help="Qdrant environment (ignored for non-Qdrant backends)")
    parser.add_argument(
        "--search-mode", default="hybrid",
        choices=["dense", "sparse", "hybrid", "hybrid_rrf"],
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--is-active", default="true", choices=["true", "false", "any"],
    )
    parser.add_argument(
        "--vector-store-id", type=int, default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help=(
            "1=Qdrant(internal) 2=Qdrant(user) "
            "3=Pinecone 4=Redis 5=PGVector 6=MongoDB"
        ),
    )
    # Individual credential flags — avoids PowerShell JSON quoting issues.
    # These are merged into vs_details after parsing.
    parser.add_argument("--vector-store-details", default=None,
                        help="JSON string (Linux/Mac). On Windows prefer the flags below.")
    parser.add_argument("--pg-host",       default=None, help="PGVector: PG_HOST")
    parser.add_argument("--pg-port",       default=None, help="PGVector: PG_PORT")
    parser.add_argument("--pg-database",   default=None, help="PGVector: PG_DATABASE")
    parser.add_argument("--pg-user",       default=None, help="PGVector: PG_USER")
    parser.add_argument("--pg-password",   default=None, help="PGVector: PG_PASSWORD")
    parser.add_argument("--qdrant-url",    default=None, help="Qdrant: QDRANT_URL")
    parser.add_argument("--qdrant-api-key",default=None, help="Qdrant: QDRANT_API_KEY")
    parser.add_argument("--pinecone-api-key",   default=None, help="Pinecone: PINECONE_API_KEY")
    parser.add_argument("--pinecone-index-name",default=None, help="Pinecone: PINECONE_INDEX_NAME")
    parser.add_argument("--pinecone-namespace", default=None, help="Pinecone: PINECONE_NAMESPACE")
    parser.add_argument("--redis-uri",      default=None, help="Redis: REDIS_URI")
    parser.add_argument("--redis-host",     default=None, help="Redis: REDIS_HOST")
    parser.add_argument("--redis-port",     default=None, help="Redis: REDIS_PORT")
    parser.add_argument("--redis-password", default=None, help="Redis: REDIS_PASSWORD")
    parser.add_argument("--mongo-uri",      default=None, help="MongoDB: MONGODB_URI")
    parser.add_argument("--mongo-database", default=None, help="MongoDB: MONGODB_DATABASE")
    parser.add_argument("--mongo-index-name",default=None,help="MongoDB: MONGODB_INDEX_NAME")
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--url",       default="http://103.207.149.22:8001",
                        help="LLM Triton server URL")
    parser.add_argument("--llm-model", default="llama")

    args = parser.parse_args()

    is_active_map    = {"true": True, "false": False, "any": None}
    is_active_filter = is_active_map[args.is_active]

    vs_details: Dict[str, Any] = {"vector_store_id": args.vector_store_id}

    # 1. JSON blob (Linux/Mac style) — strip PowerShell stray quotes if present
    if args.vector_store_details:
        try:
            raw = args.vector_store_details.strip().strip("'")
            vs_details.update(json.loads(raw))
        except json.JSONDecodeError as exc:
            parser.error(f"--vector-store-details must be valid JSON: {exc}")

    # 2. Individual flags (Windows-friendly) — each flag maps to its credential key.
    #    These overlay any value set via --vector-store-details.
    _flag_map = {
        "pg_host":            "PG_HOST",
        "pg_port":            "PG_PORT",
        "pg_database":        "PG_DATABASE",
        "pg_user":            "PG_USER",
        "pg_password":        "PG_PASSWORD",
        "qdrant_url":         "QDRANT_URL",
        "qdrant_api_key":     "QDRANT_API_KEY",
        "pinecone_api_key":   "PINECONE_API_KEY",
        "pinecone_index_name":"PINECONE_INDEX_NAME",
        "pinecone_namespace": "PINECONE_NAMESPACE",
        "redis_host":         "REDIS_HOST",
        "redis_port":         "REDIS_PORT",
        "redis_password":     "REDIS_PASSWORD",
        "mongo_uri":          "MONGODB_URI",
        "mongo_database":     "MONGODB_DATABASE",
        "mongo_index_name":   "MONGODB_INDEX_NAME",
    }
    for attr, key in _flag_map.items():
        val = getattr(args, attr, None)
        if val is not None:
            vs_details[key] = val

    backend_labels = {
        1: "Qdrant (internal)", 2: "Qdrant (user-provided)",
        3: "Pinecone", 4: "MongoDB Atlas", 5: "PGVector", 6: "Redis",
    }

    print(f"\n🚀 Multi-Backend RAG Pipeline | Mode: {args.search_mode.upper()}")
    print(f"Backend    : {backend_labels.get(args.vector_store_id, '?')} (id={args.vector_store_id})")
    print(f"Collection : {args.collection}")
    print(f"Dense      : {args.embed_model}")
    print(f"Sparse     : Remote SPLADE ({SPARSE_EMBED_URL})")
    print(f"RRF K      : {RRF_K}")
    print(f"isActive   : {args.is_active}")
    print()

    messages = [{"role": "system", "content":
        "You are a helpful assistant. Answer using only the provided context."}]
    sampling = {"max_tokens": 512, "temperature": 0.2, "top_p": 0.95}

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in {"exit", "quit"}:
            break
        if not user_input:
            continue

        try:
            chunks = retrieve(
                query                = user_input,
                collection           = args.collection,
                mode                 = args.search_mode,
                dense_model          = args.embed_model,
                env                  = args.env,
                limit                = args.top_k,
                score_threshold      = args.score_threshold,
                is_active            = is_active_filter,
                vector_store_details = vs_details,
            )
        except RetrievalExecutionError as exc:
            print(f"Retrieval failed: {exc}\n")
            continue

        if chunks:
            print("  [context injected]")
            for i, c in enumerate(chunks, 1):
                preview = c.text[:120].replace("\n", " ")
                print(f"    [{i}] score={c.score:.4f}  {preview}...")

        augmented = build_augmented_user_message(user_input, chunks)
        if not chunks:
            no_context_reply = "I could not find any information about this in the knowledge base."
            messages.append({"role": "user", "content": augmented})
            messages = truncate_history(messages, max_pairs=8)
            messages.append({"role": "assistant", "content": no_context_reply})
            print(f"Assistant> {no_context_reply}\n")
            continue

        messages.append({"role": "user", "content": augmented})
        messages = truncate_history(messages, max_pairs=8)
        prompt   = format_llama3_chat(messages)

        print("Assistant> ", end="", flush=True)
        full: List[str] = []
        for event in stream_from_triton(args.url, args.llm_model, prompt, sampling):
            token = event.get("text_output", "")
            if token:
                full.append(token)
                print(token, end="", flush=True)
        print("\n")
        messages.append({"role": "assistant", "content": "".join(full)})


# ========================= FASTAPI APP =========================

def create_app():
    """
    Build and return the FastAPI application.

    Imported lazily so that plain CLI usage does not require fastapi/uvicorn.
    """
    try:
        from fastapi import FastAPI, HTTPException, Request, status
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field, field_validator, model_validator
    except ImportError as exc:
        raise ImportError(
            "FastAPI not installed. Run: pip install fastapi uvicorn pydantic"
        ) from exc

    # ── Pydantic models ────────────────────────────────────────────────────

    class ChatTurn(BaseModel):
        """One earlier message of the conversation."""
        role: Literal["user", "assistant"]
        content: str = Field(..., min_length=1)

    class RetrieveRequest(BaseModel):
        """Retrieval request body."""

        model_config = {
            "json_schema_extra": {
                "examples": [
                    {
                        "query": "tell me about AI cloud services",
                        "collection": "knowledge_base_292003",
                        "mode": "hybrid",
                        "dense_model": "bge-large",
                        "limit": 5,
                        "score_threshold": 0.35,
                        "is_active": True,
                        "org_id": "298",
                        "role": "admin",
                        "filters": {
                            "source_type_name": "File Upload"
                        },
                        "vector_store_details": {
                            "vector_store_id": 6,
                            "redis_url": "redis://default:kTzTZh1URlhCCK5O7e4rs0ECMFk77nBM@redis-15688.c270.us-east-1-3.ec2.cloud.redislabs.com:15688"
                        }
                    }
                ]
            }
        }

        # Required
        query: Optional[str] = Field(
            default=None,
            min_length=1,
            description="Natural-language query string. Optional when point_ids are provided.",
        )
        collection: str = Field(..., min_length=1,
                                description="Collection / table / namespace name (e.g. '20034_63')")
        history: List[ChatTurn] = Field(
            default_factory=list,
            max_length=50,
            description=(
                "Earlier turns of this conversation, oldest first (alternating "
                "user / assistant). When present, a follow-up such as 'how old is "
                "he?' is rewritten into a standalone query before searching, and "
                "the conversation is passed to the answering model. Omit for a "
                "single-turn question."
            ),
        )
        rewrite_query: bool = Field(
            default=True,
            description=(
                "Rewrite the query using `history` before searching. Set false to "
                "search the query exactly as written (history is still shown to the "
                "answering model)."
            ),
        )

        # Search options
        mode: Literal["dense", "sparse", "hybrid", "hybrid_rrf"] = Field(
            default="hybrid",
            description="Search mode: dense | sparse | hybrid | hybrid_rrf",
        )
        limit:           int   = Field(default=5,    ge=1, le=100)
        score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
        dense_model:     str   = Field(default=DEFAULT_DENSE_MODEL,
                                       description="Embedding model slug")
        env:             str   = Field(default="uat",
                                       description="Qdrant env key ('dev'/'uat') — ignored for other backends")
        is_active: Optional[bool] = Field(
            default=True,
            description="Filter on isActive flag. true=active only, false=inactive only, null=no filter",
        )
        text_format: Literal["markdown", "plain"] = Field(
            default="markdown",
            description=(
                "Format of each result's `text`. 'markdown' (default) returns the "
                "stored chunk text verbatim. 'plain' flattens Markdown markup "
                "(headings, tables, labels) into a single plain-text paragraph."
            ),
        )
        openai_api_key: Optional[str] = Field(
            default=None,
            description=(
                "OpenAI API key for answer generation. When provided, it overrides "
                "the server's OPENAI_API_KEY from .env for this request only."
            ),
        )
        openai_base_url: Optional[str] = Field(
            default=None,
            description="Override the OpenAI base URL for this request (defaults to the server config).",
        )
        openai_chat_model: Optional[str] = Field(
            default=None,
            description="Override the OpenAI chat model for this request (defaults to the server config).",
        )
        vector_store_details: Dict[str, Any] = Field(
            default_factory=dict,
            description=(
                "Backend credentials. Must include 'vector_store_id'. "
                "1=Qdrant(internal) 2=Qdrant(user) 3=Pinecone "
                "4=Redis 5=PGVector 6=MongoDB Atlas"
            ),
        )
        filters: Optional[Dict[str, Any]] = Field(
            default=None,
            description=(
                "Metadata filters to narrow search results. "
                "Supports simple key-value equality, operators ($eq, $ne, $gt, $gte, $lt, $lte, $in, $nin), "
                "and combined range filters. Examples: "
                '{"source_type_name": "File Upload"}, '
                '{"org_id": "298", "source_mapping_id": "147"}, '
                '{"chunk_index": {"$gte": 0, "$lte": 10}}, '
                '{"source_type_name": {"$in": ["File Upload", "API"]}}'
            ),
            json_schema_extra={
                "examples": [
                    {"source_type_name": "File Upload"},
                    {"org_id": "298", "source_mapping_id": "147"},
                    {"chunk_index": {"$gte": 0, "$lte": 10}},
                    {"source_type_name": {"$in": ["File Upload", "API"]}},
                ]
            },
        )
        point_ids: Optional[List[str]] = Field(
            default=None,
            description="Optional point IDs to fetch directly from Pinecone.",
        )
        include_dense_values: bool = Field(
            default=False,
            description="Include stored dense vector values in the response when supported.",
        )
        include_sparse_values: bool = Field(
            default=False,
            description="Include stored sparse vector values in the response when supported.",
        )

        # Access Control
        org_id: Optional[str] = Field(
            default=None,
            description=(
                "Organization ID (tenant isolation). Required. "
                "Only chunks belonging to this org are returned. "
                "This value is stable — set from the KB owner, never changes on re-ingestion."
            ),
        )
        role: Optional[str] = Field(
            default=None,
            description=(
                "Role of the requesting user (e.g. 'admin', 'manager', 'viewer'). "
                "When provided, only chunks with matching access_role metadata are returned."
            ),
        )

        @field_validator("vector_store_details")
        @classmethod
        def validate_vector_store_id(cls, v: Dict[str, Any]) -> Dict[str, Any]:
            vsid = v.get("vector_store_id")
            if vsid is not None and int(vsid) not in {1, 2, 3, 4, 5, 6}:
                raise ValueError(
                    f"vector_store_id must be 1–6, got {vsid}. "
                    "1=Qdrant(internal) 2=Qdrant(user) 3=Pinecone "
                    "4=Redis 5=PGVector 6=MongoDB"
                )
            return v

        @model_validator(mode="after")
        def validate_query_or_point_ids(self):
            has_query = bool((self.query or "").strip())
            has_point_ids = bool(self.point_ids)
            if not has_query and not has_point_ids:
                raise ValueError("Either query or point_ids must be provided")
            return self

        model_config = {"json_schema_extra": {"examples": [{
            "query": "tell me about AI cloud services",
            "collection": "knowledge_base_292003",
            "mode": "hybrid",
            "limit": 5,
            "score_threshold": 0.35,
            "dense_model": DEFAULT_DENSE_MODEL,
            "is_active": True,
            "org_id": "298",
            "role": "admin",
            "filters": {"source_type_name": "File Upload"},
            "vector_store_details": {
                "vector_store_id": 6,
                "redis_url": "redis://default:reLRubuhTkwnGwgAUNVWE47iV05nxiZm@redis-13013.c263.us-east-1-2.ec2.cloud.redislabs.com:13013",
            },
        }, {
            "query": "tell me about Kansas Saloon Smashers",
            "collection": "20034_63",
            "mode": "hybrid",
            "limit": 5,
            "score_threshold": 0.35,
            "dense_model": DEFAULT_DENSE_MODEL,
            "is_active": True,
            "org_id": "298",
            "vector_store_details": {
                "vector_store_id": 5,
                "PG_HOST": "localhost",
                "PG_DATABASE": "ragtest",
                "PG_USER": "postgres",
                "PG_PASSWORD": "root",
            },
        }, {
            "collection": "20034_63",
            "point_ids": ["1522774727b3c489"],
            "vector_store_details": {
                "vector_store_id": 3,
                "PINECONE_API_KEY": "***",
                "PINECONE_INDEX_NAME": "rag-testing",
                "PINECONE_NAMESPACE": "20034_63",
            },
            "include_sparse_values": True,
        }]}}

    class SparseValuesResult(BaseModel):
        indices: List[int]
        values: List[float]

    class ChunkResult(BaseModel):
        point_id: Optional[str] = None
        score: float
        text: str
        metadata: Dict[str, Any] = Field(default_factory=dict)
        dense_values: Optional[List[float]] = None
        sparse_values: Optional[SparseValuesResult] = None
        version_number: Optional[int] = Field(
            default=None,
            description="Version number this chunk belongs to",
        )
        is_current_version: Optional[bool] = Field(
            default=None,
            description="Whether this chunk is from the current (active) version",
        )

    class VersionSummary(BaseModel):
        version_number: int
        timestamp: Optional[str] = None
        is_active: bool = False
        chunk_count: int = 0

    class RetrieveResponse(BaseModel):
        query:         str
        collection:    str
        mode:          str
        backend_id:    int
        total_results: int
        latency_ms:    float
        results:       List[ChunkResult]
        answer:        str = Field(..., description="Final answer generated by the LLM")
        rewritten_query: Optional[str] = Field(
            default=None,
            description="The standalone query actually searched, when a follow-up was rewritten using `history`.",
        )
        available_versions: Optional[List[VersionSummary]] = Field(
            default=None,
            description="List of available versions for this knowledge base (if versioning is enabled)",
        )

    class ErrorResponse(BaseModel):
        detail:     str
        error_type: Optional[str] = None

    # ── App ────────────────────────────────────────────────────────────────

    app = FastAPI(
        title="RAG Retrieval API",
        description=(
            "Multi-backend vector retrieval — "
            "Qdrant, Pinecone, Redis, PGVector, MongoDB Atlas"
        ),
        version="1.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def _global_exc_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled exception: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": str(exc), "error_type": type(exc).__name__},
        )

    # ── Endpoints ──────────────────────────────────────────────────────────

    @app.get("/retrieve/health", tags=["System"])
    async def health_check() -> Dict[str, str]:
        """Liveness probe."""
        return {"status": "ok", "service": "retrieval-api"}

    # =====================================================================
    # Tool-calling configuration CRUD (persistent, org-scoped).
    # These back the "saved tool-calling steps" feature: create/list/update/
    # delete reusable external-API tools that /retrieve/smart can call. Configs
    # persist to a JSON file and survive restarts. Everything is scoped by
    # org_id for tenant isolation.
    # =====================================================================
    try:
        from retrieval.tool_store import ToolConfig, get_tool_store
    except ImportError:
        from tool_store import ToolConfig, get_tool_store  # type: ignore

    class ToolConfigCreate(BaseModel):
        """Create/update payload. `id`/`org_id`/timestamps are server-managed."""
        name: str = Field(..., min_length=1, max_length=100)
        description: str = Field(..., min_length=1)
        url: str = Field(..., min_length=1)
        method: str = Field(default="GET")
        headers: Dict[str, str] = Field(default_factory=dict)
        query_params: Dict[str, Any] = Field(default_factory=dict)
        json_body: Optional[Dict[str, Any]] = Field(default=None)
        timeout: int = Field(default=20, ge=1, le=120)
        response_path: Optional[str] = Field(default=None)
        auth: Dict[str, Any] = Field(
            default_factory=lambda: {"type": "none"},
            description="Auth block, e.g. {'type':'bearer token','token':'...'}.",
        )
        pagination: Dict[str, Any] = Field(
            default_factory=lambda: {"type": "none"},
            description="Pagination block; omit or {'pagination_type':'none'} for single-call tools.",
        )

        model_config = {
            "json_schema_extra": {
                "examples": [{
                    "name": "weather",
                    "description": "Live current weather by city.",
                    "url": "https://api.weatherapi.com/v1/current.json",
                    "method": "GET",
                    "query_params": {"q": "{location}"},
                    "response_path": "current.condition.text",
                    "auth": {"type": "api key", "header_name": "key", "api_key": "YOUR_KEY"}
                }]
            }
        }

    def _build_tool_config(org_id: str, payload: "ToolConfigCreate") -> ToolConfig:
        """Validate a create/update payload into a full ToolConfig."""
        data = payload.model_dump()
        # Normalize an empty/absent pagination block to the 'none' default.
        pg = data.get("pagination") or {}
        if not pg or pg == {"type": "none"}:
            data["pagination"] = {"pagination_type": "none"}
        data["org_id"] = str(org_id)
        try:
            return ToolConfig(**data)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"Invalid tool configuration: {exc}",
            )

    @app.post("/tools", tags=["Tools"], summary="Create a tool-calling configuration")
    async def create_tool(payload: ToolConfigCreate, org_id: str) -> Dict[str, Any]:
        """Save a new tool config for an org. `org_id` is a required query param."""
        if not org_id:
            raise HTTPException(status_code=400, detail="org_id is required.")
        config = _build_tool_config(org_id, payload)
        try:
            created = get_tool_store().create(config)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
        return json.loads(created.model_dump_json())

    @app.get("/tools", tags=["Tools"], summary="List tool-calling configurations")
    async def list_tools(org_id: str) -> Dict[str, Any]:
        """List all tool configs for an org. `org_id` is a required query param."""
        if not org_id:
            raise HTTPException(status_code=400, detail="org_id is required.")
        tools = get_tool_store().list(str(org_id))
        return {
            "org_id": str(org_id),
            "count": len(tools),
            "tools": [json.loads(t.model_dump_json()) for t in tools],
        }

    @app.get("/tools/{tool_id}", tags=["Tools"], summary="Get one tool-calling configuration")
    async def get_tool(tool_id: str, org_id: str) -> Dict[str, Any]:
        if not org_id:
            raise HTTPException(status_code=400, detail="org_id is required.")
        cfg = get_tool_store().get(str(org_id), tool_id)
        if cfg is None:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found.")
        return json.loads(cfg.model_dump_json())

    @app.put("/tools/{tool_id}", tags=["Tools"], summary="Update a tool-calling configuration")
    async def update_tool(tool_id: str, payload: ToolConfigCreate, org_id: str) -> Dict[str, Any]:
        if not org_id:
            raise HTTPException(status_code=400, detail="org_id is required.")
        config = _build_tool_config(org_id, payload)
        try:
            updated = get_tool_store().update(str(org_id), tool_id, config)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
        if updated is None:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found.")
        return json.loads(updated.model_dump_json())

    @app.delete("/tools/{tool_id}", tags=["Tools"], summary="Delete a tool-calling configuration")
    async def delete_tool(tool_id: str, org_id: str) -> Dict[str, Any]:
        if not org_id:
            raise HTTPException(status_code=400, detail="org_id is required.")
        removed = get_tool_store().delete(str(org_id), tool_id)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Tool '{tool_id}' not found.")
        return {"status": "deleted", "id": tool_id, "org_id": str(org_id)}



    @app.post(
        "/retrieve/mongo-debug",
        tags=["System"],
        summary="Diagnose a MongoDB Atlas collection and vector search index",
    )
    async def mongo_debug(body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inspect a MongoDB collection without running a real vector search.

        Required body fields:
          - collection       : collection name (e.g. "218_53")
          - vector_store_details : must contain MONGODB_URI and MONGODB_DATABASE

        Returns doc count, a sample document's top-level keys, and
        whether the configured vector search index exists.
        """
        try:
            from pymongo import MongoClient
        except ImportError:
            raise HTTPException(status_code=500, detail="pip install pymongo")

        vs_details = body.get("vector_store_details", {})
        collection = body.get("collection", "")
        uri        = vs_details.get("MONGODB_URI") or vs_details.get("mongodb_uri")
        database   = vs_details.get("MONGODB_DATABASE") or vs_details.get("mongodb_database")
        index_name = vs_details.get("MONGODB_INDEX_NAME") or vs_details.get("mongodb_index_name") or "vector_index"
        dense_field = vs_details.get("MONGODB_DENSE_FIELD") or vs_details.get("mongodb_dense_field") or "dense"

        if not uri or not database:
            raise HTTPException(status_code=400, detail="MONGODB_URI and MONGODB_DATABASE are required")

        info: Dict[str, Any] = {
            "database": database,
            "collection": collection,
            "configured_index_name": index_name,
            "configured_dense_field": dense_field,
        }

        try:
            client = _create_mongo_atlas_client(uri, vs_details=vs_details)
            col    = client[database][collection]

            # Document count
            info["doc_count"] = col.count_documents({})

            # Sample document keys
            sample = col.find_one({})
            if sample:
                info["sample_doc_keys"] = list(sample.keys())
                # Check whether dense_field exists and what type it is
                fv = sample.get(dense_field)
                info["dense_field_exists_in_sample"] = fv is not None
                if isinstance(fv, list):
                    info["dense_field_length"] = len(fv)
                elif isinstance(fv, str):
                    info["dense_field_note"] = "stored as string — may need parsing"
            else:
                info["sample_doc_keys"] = []
                info["dense_field_exists_in_sample"] = False

            # List Atlas Search indexes on this collection
            try:
                indexes = list(col.list_search_indexes())
                info["search_indexes"] = [
                    {"name": idx.get("name"), "status": idx.get("status"), "type": idx.get("type")}
                    for idx in indexes
                ]
                configured_idx_names = [idx.get("name") for idx in indexes]
                info["configured_index_found"] = index_name in configured_idx_names
            except Exception as idx_exc:
                info["search_indexes_error"] = str(idx_exc)
                info["configured_index_found"] = None

            client.close()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

        return info

    @app.post(
        "/retrieve",
        response_model=RetrieveResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Invalid request"},
            500: {"model": ErrorResponse, "description": "Retrieval error"},
        },
        tags=["Retrieval"],
        summary="Retrieve chunks + generate final LLM answer",
    )
    async def retrieve_chunks(body: RetrieveRequest) -> RetrieveResponse:
        """
        Retrieve relevant chunks AND generate the final answer using the LLM.

        Access Control:
            - org_id is mandatory (tenant isolation). Only chunks belonging
              to the caller's org are returned.
            - access_role is optional — filters chunks that have access_role metadata.
        """
        # ── Auth: enforce org_id ───────────────────────────────────────────
        if not body.org_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="org_id is required for retrieval (tenant isolation).",
            )

        # ── Build access-control filters ───────────────────────────────────
        # org_id is always enforced server-side. Cannot be bypassed.
        # member_id is NOT a filter — it's audit only.
        # access_role is optional (for role-gated KBs).
        access_filters = {"org_id": body.org_id}
        if body.role:
            access_filters["access_role"] = body.role

        # Merge into user-provided filters (access filters take precedence)
        effective_filters = dict(body.filters or {})
        effective_filters.update(access_filters)

        vs_details = body.vector_store_details
        backend_id = int(vs_details.get("vector_store_id") or 1)

        logger.info(
            "Retrieve request | query=%r | collection=%s | mode=%s | backend=%d | org_id=%s",
            body.query[:80] if body.query else None, body.collection, body.mode, backend_id,
            body.org_id,
        )

        start = time.perf_counter()

        try:
            # 0. Follow-up handling: turn "how old is he?" into a standalone
            #    search query using the earlier turns (no-op without history).
            conversation = _history_messages(body.history)
            search_query = body.query or ""
            if conversation and body.rewrite_query and search_query.strip():
                search_query = rewrite_followup_query(
                    search_query,
                    conversation,
                    generate_fn=lambda msgs: _generate_answer_gpt(
                        msgs,
                        api_key=body.openai_api_key,
                        base_url=body.openai_base_url,
                        model=body.openai_chat_model,
                    ),
                )
                if search_query != (body.query or ""):
                    logger.info("Follow-up rewritten | %r -> %r", body.query, search_query)
            from_history = search_query == NO_SEARCH
            if from_history:
                search_query = body.query or ""

            # 1. Retrieval (skipped when the user only wants the earlier answer reshaped)
            raw_results: List[RetrievalResult] = [] if from_history else retrieve(
                query                = search_query,
                collection           = body.collection,
                mode                 = body.mode,
                dense_model          = body.dense_model,
                env                  = body.env,
                limit                = body.limit,
                score_threshold      = body.score_threshold,
                is_active            = body.is_active,
                vector_store_details = vs_details,
                filters              = effective_filters,
                version              = None,
                point_ids            = body.point_ids,
                include_dense_values = body.include_dense_values,
                include_sparse_values= body.include_sparse_values,
            )

            # 2. Build augmented prompt (the user's own wording; the earlier
            #    conversation goes in as real chat turns before it)
            augmented = (
                (body.query or "") if from_history
                else build_augmented_user_message(body.query or "", raw_results)
            )

            # 3. Generate final answer using GPT-4.1
            system_prompt = "You are a helpful assistant. Answer using only the provided context."
            if from_history:
                system_prompt = _RESHAPE_SYSTEM_PROMPT
            else:
                if conversation:
                    system_prompt += _HISTORY_SYSTEM_NOTE
                system_prompt += _ANSWER_STYLE_NOTE
            messages = (
                [{"role": "system", "content": system_prompt}]
                + (conversation if from_history else _deformat_history(conversation))
                + [{"role": "user", "content": augmented}]
            )

            final_answer = _generate_answer_gpt(
                messages,
                api_key=body.openai_api_key,
                base_url=body.openai_base_url,
                model=body.openai_chat_model,
            )

        except Exception as exc:
            logger.error("Retrieval error: %s", exc, exc_info=True)
            if any(
                getattr(getattr(e, "response", None), "status_code", None) == 404
                or getattr(e, "pgcode", None) == "42P01"   # Postgres: undefined table
                for e in (exc, exc.__cause__)
            ):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Collection '{body.collection}' was not found.",
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=_public_error(exc),
            )

        latency_ms = (time.perf_counter() - start) * 1000

        as_plain = body.text_format == "plain"
        chunks = [
            ChunkResult(
                point_id      = r.point_id,
                score         = round(r.score, 6),
                text          = _to_plain_text(r.text) if as_plain else r.text,
                metadata      = r.metadata,
                dense_values  = r.dense_values,
                sparse_values = r.sparse_values,
                version_number     = r.metadata.get("version_number"),
                is_current_version = r.metadata.get("is_current_version"),
            )
            for r in raw_results
        ]

        # Fetch available versions for this knowledge base
        available_versions = None
        try:
            try:
                from retrieval._vendor.version_service import get_available_versions
            except ImportError:
                try:
                    from _vendor.version_service import get_available_versions  # type: ignore
                except ImportError:
                    from services.version_service import get_available_versions  # type: ignore
            try:
                from retrieval.vectore_store import get_vector_store as _get_vs  # type: ignore
            except ImportError:
                from services.vectore_store import get_vector_store as _get_vs  # type: ignore
            _vs_for_versions = _get_vs(
                collection_name=body.collection,
                vector_size=0,
                request_data={"vector_store_details": vs_details},
                ensure_collection=False,
            )
            _versions_list = get_available_versions(_vs_for_versions, body.collection)
            if _versions_list:
                available_versions = [
                    VersionSummary(
                        version_number=v["version_number"],
                        is_active=v.get("is_active", False),
                    )
                    for v in _versions_list
                ]
        except Exception:
            pass  # Non-critical — don't fail retrieval if version lookup fails

        logger.info(
            "Retrieve done | collection=%s | backend=%d | results=%d | latency=%.1fms",
            body.collection, backend_id, len(chunks), latency_ms,
        )

        return RetrieveResponse(
            query         = body.query or "",
            collection    = body.collection,
            mode          = body.mode,
            backend_id    = backend_id,
            total_results = len(chunks),
            latency_ms    = round(latency_ms, 2),
            results       = chunks,
            answer        = final_answer or "I could not generate an answer.",
            rewritten_query = search_query if search_query != (body.query or "") else None,
            available_versions = available_versions,
        )

    # =====================================================================
    # Smart routing: decide RAG vs external API per query, then answer.
    # =====================================================================
    try:
        # When imported as part of the package: `retrieval.retrieval_new`
        from retrieval.query_router import (
            RouteSource,
            decide_route,
            registry_from_api_specs,
        )
    except ImportError:
        # When the folder itself is on sys.path (e.g. --app-dir retrieval)
        from query_router import (
            RouteSource,
            decide_route,
            registry_from_api_specs,
        )

    class ApiSpecModel(BaseModel):
        """One external API the UI supplies for this request.

        String values in url/headers/query_params/json_body may use
        {placeholders} (e.g. {query}, {location}) filled from the router's
        extracted parameters.
        """
        name: str = Field(..., description="Unique tool name the router selects (e.g. 'weather').")
        description: str = Field(..., description="What this API returns. The router uses this to pick it.")
        url: str = Field(default="", description="Full API URL for http tools. May contain {placeholders}.")
        kind: Optional[str] = Field(
            default=None,
            description=(
                "Connector kind: 'http' (default), 'database', 'api', or "
                "'cloud_storage'. Inferred from the shape when omitted (e.g. a "
                "db_credentials/config block implies 'database')."
            ),
        )
        config: Dict[str, Any] = Field(
            default_factory=dict,
            description=(
                "Per-service config for non-http tools (the source_details the "
                "existing DatabaseSource/APISource/CloudStorageSource accept). "
                "Supports {placeholders}."
            ),
        )
        db_credentials: Optional[Dict[str, Any]] = Field(
            default=None,
            description="Convenience alias: DB connection fields merged into config for 'database' tools.",
        )
        method: str = Field(default="GET", description="HTTP method: GET | POST | ...")
        headers: Dict[str, str] = Field(default_factory=dict, description="Request headers (supports {placeholders}).")
        query_params: Dict[str, Any] = Field(default_factory=dict, description="Query string params (supports {placeholders}).")
        json_body: Optional[Dict[str, Any]] = Field(default=None, description="JSON body for POST/PUT (supports {placeholders}).")
        timeout: int = Field(default=20, ge=1, le=120)
        response_path: Optional[str] = Field(default=None, description="Dotted path into the JSON response to extract, e.g. 'current.temp_c'.")
        auth: Optional[Dict[str, Any]] = Field(
            default=None,
            description=(
                "Optional structured authentication, applied to this call's headers. "
                "Shapes: {'type':'none'}, {'type':'basic','username':..,'password':..}, "
                "{'type':'bearer token','token':..}, {'type':'api key','header_name':..,'api_key':..}. "
                "Any Authorization/header you set manually in `headers` wins over this."
            ),
        )
        pagination: Optional[Dict[str, Any]] = Field(
            default=None,
            description=(
                "Optional first-page pagination, seeded into this call's query params. "
                "Shape: {'pagination_type':'page_number','page_param_name':'page',"
                "'page_size_param_name':'size','page_size_value':20,'start_page':1}. "
                "Omit or {'pagination_type':'none'} for single-call tools."
            ),
        )

        model_config = {
            "json_schema_extra": {
                "examples": [{
                    "name": "weather",
                    "description": "Live current weather by city. Use for weather/temperature/forecast questions.",
                    "url": "https://api.weatherapi.com/v1/current.json",
                    "method": "GET",
                    "query_params": {"key": "YOUR_WEATHER_KEY", "q": "{location}"},
                    "response_path": "current.condition.text"
                }]
            }
        }

    class SmartRetrieveRequest(RetrieveRequest):
        """Retrieve request plus routing controls and per-request external APIs."""
        apis: List[ApiSpecModel] = Field(
            default_factory=list,
            description=(
                "External APIs available for this query (supplied by the UI). The "
                "router classifies the query and, if it needs live data, picks the "
                "best-matching API by its description and calls it. If the query "
                "needs stored documents, the knowledge base (RAG) is used instead."
            ),
        )
        use_llm_router: bool = Field(
            default=True,
            description="Use the LLM to decide RAG vs API. Falls back to heuristics on failure.",
        )
        use_saved_tools: bool = Field(
            default=True,
            description=(
                "Also consider tool-calling configurations saved for this org via "
                "the /tools endpoints. Per-request `apis` override a saved tool of "
                "the same name. Set false to use only the per-request `apis`."
            ),
        )
        force_source: Optional[Literal["rag", "api", "hybrid"]] = Field(
            default=None,
            description="Override the router and force a source. Omit for automatic routing.",
        )

    class RouteInfo(BaseModel):
        source: str
        tool: Optional[str] = None
        parameters: Dict[str, Any] = Field(default_factory=dict)
        reason: str = ""
        confidence: float = 0.0
        decided_by: str = "heuristic"

    class SmartRetrieveResponse(BaseModel):
        query: str
        rewritten_query: Optional[str] = Field(
            default=None,
            description="The standalone query actually used for routing and search, when a follow-up was rewritten using `history`.",
        )
        answer: str
        source_used: str = Field(..., description="rag | api | hybrid | history — the source actually used (history: the earlier answer was reshaped, nothing looked up)")
        route: RouteInfo
        latency_ms: float
        rag_results: List[ChunkResult] = Field(default_factory=list)
        external_data: Optional[Dict[str, Any]] = None
        fallback_used: bool = Field(
            default=False,
            description="True when the planned source failed and the system fell back to another.",
        )
        warnings: List[str] = Field(default_factory=list)

    def _run_rag(
        body: SmartRetrieveRequest,
        access_filters: Dict[str, Any],
        search_query: Optional[str] = None,
    ) -> Tuple[List[ChunkResult], str]:
        """Execute RAG retrieval + answer. Returns (chunks, context_text).

        ``search_query`` is the standalone query to search with (a follow-up
        rewritten using the conversation); it defaults to the user's query.
        """
        q = (search_query or body.query or "")
        effective_filters = dict(body.filters or {})
        effective_filters.update(access_filters)
        vs_details = body.vector_store_details

        # Over-fetch a wider candidate pool so re-ranking can surface the
        # correct chunk even when it ranked outside `limit`. This does NOT
        # raise the user-facing result count — we trim back to `body.limit`
        # after re-ranking. For exact-ID lookups (e.g. "EMP0001") dense search
        # ranks the target record arbitrarily among many near-identical rows,
        # so we fetch a much larger pool to be sure the record is included.
        has_identifier = bool(_detect_identifiers(q))
        fetch_limit = max(body.limit * 20, 200) if has_identifier else max(body.limit * 5, 25)
        raw_results: List[RetrievalResult] = retrieve(
            query                = q,
            collection           = body.collection,
            mode                 = body.mode,
            dense_model          = body.dense_model,
            env                  = body.env,
            limit                = fetch_limit,
            score_threshold      = body.score_threshold,
            is_active            = body.is_active,
            vector_store_details = vs_details,
            filters              = effective_filters,
            version              = None,
            point_ids            = body.point_ids,
        )

        # Player/entity-aware re-rank: if the query names a known player, keep
        # that player's chunks first so another player's near-identical stat
        # table cannot be returned instead. Then trim to the requested limit.
        raw_results = _rerank_by_entity(q, raw_results)
        raw_results = raw_results[: body.limit]

        as_plain = body.text_format == "plain"
        chunks = [
            ChunkResult(
                point_id      = r.point_id,
                score         = round(r.score, 6),
                text          = _to_plain_text(r.text) if as_plain else r.text,
                metadata      = r.metadata,
                version_number     = r.metadata.get("version_number"),
                is_current_version = r.metadata.get("is_current_version"),
            )
            for r in raw_results
        ]

        # Hierarchical / small-to-big expansion: we match on precise CHILD
        # chunks for ranking, but ingestion stores the broader PARENT passage
        # inside each child's `context_text`. Feed that fuller context to the
        # LLM when present (deduped by parent so a parent appears once), falling
        # back to the child `text` for non-hierarchical collections.
        expanded: List[RetrievalResult] = []
        seen_parents: set = set()
        for r in raw_results:
            md = r.metadata or {}
            parent_ctx = md.get("context_text")
            parent_id = md.get("parent_chunk_id")
            if parent_ctx:
                if parent_id is not None:
                    if parent_id in seen_parents:
                        continue  # parent context already included
                    seen_parents.add(parent_id)
                expanded.append(
                    RetrievalResult(
                        score=r.score,
                        text=parent_ctx,          # broader parent passage
                        metadata=md,
                        point_id=r.point_id,
                    )
                )
            else:
                expanded.append(r)              # non-hierarchical: use child text

        context_text = build_augmented_user_message(body.query or "", expanded)
        return chunks, context_text

    @app.post(
        "/retrieve/smart",
        response_model=SmartRetrieveResponse,
        tags=["Retrieval"],
        summary="Auto-route a query to RAG or an external API, then answer",
    )
    async def smart_retrieve(body: SmartRetrieveRequest) -> SmartRetrieveResponse:
        """Analyze the query, choose RAG / external API / hybrid, and answer.

        Error handling: each path is guarded; if the chosen external API fails
        (or no tool is available), the system falls back to RAG so the user
        still gets an answer.
        """
        if not body.org_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="org_id is required for retrieval (tenant isolation).",
            )

        access_filters: Dict[str, Any] = {"org_id": body.org_id}
        if body.role:
            access_filters["access_role"] = body.role

        warnings: List[str] = []
        start = time.perf_counter()

        # Per-request OpenAI credentials (fall back to .env when omitted).
        def _answer(messages: List[Dict[str, str]]) -> str:
            return _generate_answer_gpt(
                messages,
                api_key=body.openai_api_key,
                base_url=body.openai_base_url,
                model=body.openai_chat_model,
            )

        # Follow-up handling: rewrite "how old is he?" into a standalone query
        # using the earlier turns. Used for routing, API parameters and search;
        # the user's own wording and the real conversation go to the final answer.
        conversation = _history_messages(body.history)
        search_query = body.query or ""
        if conversation and body.rewrite_query and search_query.strip():
            search_query = rewrite_followup_query(search_query, conversation, generate_fn=_answer)
            if search_query != (body.query or ""):
                logger.info("Follow-up rewritten | %r -> %r", body.query, search_query)
        from_history = search_query == NO_SEARCH
        if from_history:
            search_query = body.query or ""
        rewritten = search_query if search_query != (body.query or "") else None

        # Build a per-request external-API registry from the APIs the UI sent.
        # Each request can carry its own set of APIs; nothing is hardcoded.
        # Inline `auth` / `pagination` blocks are applied to the spec's
        # headers / query_params using the SAME helpers as saved tools, so an
        # inline API and a saved tool authenticate/paginate identically.
        try:
            from retrieval.tool_store import (
                AuthConfig as _AuthConfig,
                PaginationConfig as _PaginationConfig,
                _apply_auth as _apply_tool_auth,
                _apply_pagination as _apply_tool_pagination,
            )
        except ImportError:
            from tool_store import (  # type: ignore
                AuthConfig as _AuthConfig,
                PaginationConfig as _PaginationConfig,
                _apply_auth as _apply_tool_auth,
                _apply_pagination as _apply_tool_pagination,
            )
        from pydantic import TypeAdapter as _TypeAdapter

        api_specs: List[Dict[str, Any]] = []
        for a in body.apis:
            spec = a.model_dump()
            inline_auth = spec.pop("auth", None)
            inline_pagination = spec.pop("pagination", None)

            # Determine the effective connector kind the SAME way
            # ApiToolSpec.from_dict does, so non-http tools (database / api /
            # cloud_storage) skip the HTTP-only processing below (JSON-Path
            # body shim, header auth, query-param pagination) which would be
            # meaningless or corrupting for them. Their `kind`/`config`/
            # `db_credentials` pass straight through to registry_from_api_specs.
            _kind = str(spec.get("kind") or spec.get("type") or "").strip().lower()
            if not _kind:
                _cfg = dict(spec.get("config") or {})
                _dbc = dict(spec.get("db_credentials") or {})
                if _dbc or _cfg.get("db_type") or _cfg.get("database_type_name") \
                        or _cfg.get("host") or _cfg.get("connection_uri"):
                    _kind = "database"
                elif _cfg.get("provider") or _cfg.get("bucket"):
                    _kind = "cloud_storage"
                elif spec.get("url"):
                    _kind = "http"
                else:
                    _kind = "http"
            if _kind != "http":
                # Non-http: drop the empty http-only key defaults so the spec
                # stays clean, keep kind/config/db_credentials, pass through.
                spec["kind"] = _kind
                api_specs.append(spec)
                continue

            headers = dict(spec.get("headers") or {})
            query_params = dict(spec.get("query_params") or {})

            # JSON-Path-as-body compatibility shim. The Step-6 UI has no
            # dedicated Request Body field, so a user may paste the request
            # body (or a {placeholder} template) into the JSON Path field.
            # When response_path looks like a JSON object/array (starts with
            # '{' or '['), treat it as the json_body and clear response_path so
            # response extraction returns the whole response. A dotted path
            # (e.g. 'data', '$.data') never starts with '{' so it is unaffected.
            rp = spec.get("response_path")
            if isinstance(rp, str) and rp.strip().startswith(("{", "[")):
                try:
                    parsed_body = json.loads(rp)
                    if isinstance(parsed_body, dict):
                        # Only set if the user didn't also send a real json_body.
                        if not spec.get("json_body"):
                            spec["json_body"] = parsed_body
                        spec["response_path"] = None
                    else:
                        warnings.append(
                            f"API '{spec.get('name')}': JSON Path body must be a JSON object; ignored."
                        )
                        spec["response_path"] = None
                except Exception as exc:
                    warnings.append(
                        f"API '{spec.get('name')}': could not parse JSON Path as a body ({exc})."
                    )

            # Apply structured auth (basic/bearer/api_key) -> headers.
            if inline_auth and str(inline_auth.get("type", "none")).lower() != "none":
                try:
                    parsed_auth = _TypeAdapter(_AuthConfig).validate_python(inline_auth)
                    _apply_tool_auth(parsed_auth, headers, query_params)
                except Exception as exc:
                    warnings.append(f"Ignored invalid auth for API '{spec.get('name')}': {exc}")

            # Apply first-page pagination -> query_params.
            pg = inline_pagination or {}
            pg_type = str(pg.get("pagination_type") or pg.get("type") or "none").lower()
            if pg and pg_type != "none":
                try:
                    norm = dict(pg)
                    norm.pop("type", None)
                    norm.setdefault("pagination_type", pg_type)
                    parsed_pg = _PaginationConfig(**norm)
                    _apply_tool_pagination(parsed_pg, query_params)
                except Exception as exc:
                    warnings.append(f"Ignored invalid pagination for API '{spec.get('name')}': {exc}")

            spec["headers"] = headers
            spec["query_params"] = query_params
            api_specs.append(spec)

        # Merge persisted tool configs saved for this org (via the /tools CRUD
        # endpoints) so they are available during retrieval without being
        # re-sent on every request. Per-request `apis` win over a saved tool of
        # the same name (name match is case-insensitive). Opt out per request
        # with use_saved_tools=false.
        if body.use_saved_tools:
            try:
                from retrieval.tool_store import get_tool_store, to_api_spec_dict
            except ImportError:
                from tool_store import get_tool_store, to_api_spec_dict  # type: ignore
            request_names = {
                str(s.get("name", "")).strip().lower() for s in api_specs
            }
            for saved in get_tool_store().list(str(body.org_id)):
                if saved.name.strip().lower() in request_names:
                    continue  # per-request API overrides the saved one
                try:
                    api_specs.append(to_api_spec_dict(saved))
                except Exception as exc:  # one bad saved tool must not break retrieval
                    warnings.append(
                        f"Skipped saved tool '{saved.name}': {exc}"
                    )

        external_registry, spec_warnings = registry_from_api_specs(api_specs)
        warnings.extend(spec_warnings)

        # 1. Decide the route (optionally forced). A message that only reshapes
        #    the earlier answer needs no lookup at all.
        if from_history:
            decision = decide_route(search_query, external_registry, use_llm=False)
            decision.source = RouteSource.RAG
            decision.tool = None
            decision.parameters = {}
            decision.reason = "Reshaping the earlier answer; no new lookup needed."
            decision.decided_by = "history"
        elif body.force_source:
            decision = decide_route(search_query, external_registry, use_llm=False)
            decision.source = RouteSource(body.force_source)
            decision.reason = f"Forced by request to '{body.force_source}'."
            decision.decided_by = "forced"
            if decision.source in (RouteSource.API, RouteSource.HYBRID) and not external_registry.has(decision.tool):
                decision.tool = external_registry.names()[0] if external_registry.names() else None
        else:
            decision = decide_route(
                search_query,
                external_registry,
                generate_fn=_answer,
                use_llm=body.use_llm_router,
            )

        # Guarantee the query is available for {placeholder} templating in API
        # tools. The heuristic router (used when forcing a source) returns a RAG
        # decision with no parameters, so forcing source=api would otherwise send
        # an empty {query}. Seed common aliases without overwriting any value the
        # LLM router already extracted.
        if decision.source in (RouteSource.API, RouteSource.HYBRID):
            if not isinstance(decision.parameters, dict):
                decision.parameters = {}
            decision.parameters.setdefault("query", search_query)
            decision.parameters.setdefault("q", search_query)

            # Query-driven field extraction: fill the chosen tool's
            # {placeholders} (in url / headers / query_params / json_body) from
            # the user query — via embedded JSON and/or an LLM extraction pass.
            # This lets a natural-language query drive create/update/delete
            # bodies and header values without the user typing raw JSON.
            try:
                from retrieval.query_router import extract_tool_parameters as _extract_params
            except ImportError:
                from query_router import extract_tool_parameters as _extract_params  # type: ignore
            chosen_spec = external_registry.get_spec(decision.tool)
            if chosen_spec is not None:
                extracted = _extract_params(
                    body.query or "", chosen_spec, generate_fn=_answer,
                )
                for k, v in extracted.items():
                    # Don't clobber values the router already extracted.
                    decision.parameters.setdefault(k, v)

        logger.info(
            "Smart route | query=%r | source=%s | tool=%s | by=%s",
            (body.query or "")[:80], decision.source.value, decision.tool, decision.decided_by,
        )

        rag_chunks: List[ChunkResult] = []
        external_data: Optional[Dict[str, Any]] = None
        external_text: str = ""
        source_used = "history" if from_history else decision.source.value
        fallback_used = False

        # 2. Gather context from the chosen source(s), with fallback.
        need_rag = decision.source in (RouteSource.RAG, RouteSource.HYBRID) and not from_history
        need_api = decision.source in (RouteSource.API, RouteSource.HYBRID) and not from_history

        rag_context = ""
        if need_api:
            tool_result = external_registry.call(decision.tool or "", decision.parameters)
            if tool_result.ok:
                external_text = tool_result.text
                external_data = {"tool": tool_result.tool, "data": tool_result.data}
            else:
                warnings.append(
                    f"External API '{decision.tool}' failed: {tool_result.error}. "
                    "Falling back to the knowledge base."
                )
                fallback_used = True
                need_rag = True  # fall back to RAG
                if decision.source == RouteSource.API:
                    source_used = "rag"
                else:
                    source_used = "hybrid"

        if need_rag:
            try:
                rag_chunks, rag_context = _run_rag(body, access_filters, search_query)
            except Exception as exc:
                logger.error("Smart RAG step failed: %s", exc, exc_info=True)
                warnings.append(f"Knowledge-base retrieval failed: {_public_error(exc)}")
                if source_used == "rag" and not external_text:
                    # Nothing to answer from.
                    latency_ms = (time.perf_counter() - start) * 1000
                    return SmartRetrieveResponse(
                        query=body.query or "",
                        rewritten_query=rewritten,
                        answer="I could not retrieve an answer from the knowledge base or external sources.",
                        source_used=source_used,
                        route=RouteInfo(**decision.to_dict()),
                        latency_ms=round(latency_ms, 2),
                        rag_results=[],
                        external_data=external_data,
                        fallback_used=fallback_used,
                        warnings=warnings,
                    )

        # 3. Build the final prompt from whatever context we have.
        context_blocks: List[str] = []
        if external_text:
            # The external block is the AUTHORITATIVE result of actually calling
            # the tool (an API response, a DB query result, a created/updated
            # record, ...). Label it as such so the model reports it instead of
            # explaining how one *would* do it.
            _tool_name = (external_data or {}).get("tool") or decision.tool or "the tool"
            context_blocks.append(
                f"--- TOOL RESULT (authoritative; what '{_tool_name}' actually "
                f"returned when it ran) ---\n{external_text}"
            )
        if rag_context:
            context_blocks.append(rag_context)

        # Pick the system prompt based on what we have. When a tool actually
        # ran and returned data, force the model to answer FROM that result and
        # report it — not to produce a generic how-to. Otherwise fall back to
        # the normal grounded-assistant behavior.
        if external_text:
            system_prompt = (
                "You are an assistant that reports the outcome of a tool/API/database "
                "call that has ALREADY been executed. The TOOL RESULT block is the "
                "real, authoritative output of that call. Answer the user strictly "
                "from that result: state what happened and surface the concrete "
                "values it contains (e.g. ids, names, fields, rows, counts, status). "
                "Do NOT give generic instructions, example payloads, or cURL/code "
                "snippets, and do NOT tell the user how they could perform the action "
                "— it has already been performed. If the result indicates an error or "
                "is empty, say so plainly. Be concise and factual."
            )
        else:
            system_prompt = (
                "You are a helpful assistant. Use the provided context when available."
            )

        if context_blocks:
            user_content = (
                "Answer the user's question using the context below. The TOOL RESULT "
                "block (when present) is the authoritative output of a call that has "
                "already run — report it and its concrete values. Prefer live tool "
                "data for live/action facts and the knowledge base for stored/document "
                "facts.\n\n"
                + "\n\n".join(context_blocks)
                + f"\n\nQuestion: {body.query}"
            )
        else:
            user_content = body.query or ""

        try:
            if from_history:
                system_prompt = _RESHAPE_SYSTEM_PROMPT
            else:
                if conversation:
                    system_prompt += _HISTORY_SYSTEM_NOTE
                system_prompt += _ANSWER_STYLE_NOTE
            answer = _answer(
                [{"role": "system", "content": system_prompt}]
                + (conversation if from_history else _deformat_history(conversation))
                + [{"role": "user", "content": user_content}]
            )
        except Exception as exc:
            logger.error("Smart answer generation failed: %s", exc, exc_info=True)
            answer = "I found relevant information but could not generate a final answer."
            warnings.append(f"Answer generation failed: {exc}")

        latency_ms = (time.perf_counter() - start) * 1000
        return SmartRetrieveResponse(
            query=body.query or "",
            rewritten_query=rewritten,
            answer=answer or "I could not generate an answer.",
            source_used=source_used,
            route=RouteInfo(**decision.to_dict()),
            latency_ms=round(latency_ms, 2),
            rag_results=rag_chunks,
            external_data=external_data,
            fallback_used=fallback_used,
            warnings=warnings,
        )

    return app


# Expose module-level `app` so  uvicorn retrieval_new:app  works directly
app = create_app()


if __name__ == "__main__":
    main()