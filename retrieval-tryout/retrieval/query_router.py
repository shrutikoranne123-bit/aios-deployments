"""
query_router.py — Query decision + source routing for the Knowledge Base.
==========================================================================

Analyzes each user query and decides whether to answer from:

  - RAG     : retrieve from ingested documents / the vector knowledge base
  - API     : call an external API for live / dynamic data
  - HYBRID  : combine knowledge-base context with live external data

Decision strategy (two tiers, with graceful fallback):

  1. LLM router (primary)  — a small structured classification call to the
     same OpenAI model used for answers. Returns the route, the external tool
     to use (if any), and extracted parameters.
  2. Heuristic router (fallback) — keyword / pattern rules used when the LLM
     router is unavailable (no API key, timeout, or malformed output), so the
     system always makes a reasonable decision.

External APIs are pluggable via `ExternalAPIRegistry`. Register real tools
(weather, web search, stock quotes, ...) without touching the router. If no
tool matches or a call fails, the pipeline falls back to RAG.

This module is intentionally free of FastAPI imports so it can be unit-tested
and reused as a plain library. The HTTP endpoint lives in retrieval_new.py.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ===========================================================================
# Decision types
# ===========================================================================

class RouteSource(str, Enum):
    """Where the answer should come from."""
    RAG = "rag"
    API = "api"
    HYBRID = "hybrid"


@dataclass
class RouteDecision:
    """The router's decision for a single query."""
    source: RouteSource
    tool: Optional[str] = None               # external tool name, when source != RAG
    parameters: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    confidence: float = 0.0
    decided_by: str = "heuristic"            # "llm" | "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.value,
            "tool": self.tool,
            "parameters": self.parameters,
            "reason": self.reason,
            "confidence": round(self.confidence, 3),
            "decided_by": self.decided_by,
        }


# ===========================================================================
# External API layer (pluggable)
# ===========================================================================

@dataclass
class ExternalToolResult:
    """Normalized result returned by any external tool."""
    ok: bool
    text: str = ""
    data: Any = None
    error: Optional[str] = None
    tool: str = ""


# A tool is any callable: (parameters: dict) -> ExternalToolResult
ExternalTool = Callable[[Dict[str, Any]], ExternalToolResult]


class ExternalAPIRegistry:
    """Registry mapping tool names to callables and their descriptions.

    Descriptions are fed to the LLM router so it can pick the right tool.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, ExternalTool] = {}
        self._descriptions: Dict[str, str] = {}
        # Optional: the ApiToolSpec behind each tool (when built from specs).
        # Lets callers inspect placeholders / do query-driven extraction.
        self._specs: Dict[str, "ApiToolSpec"] = {}

    def register(
        self,
        name: str,
        tool: ExternalTool,
        description: str,
        spec: Optional["ApiToolSpec"] = None,
    ) -> None:
        key = name.strip().lower()
        self._tools[key] = tool
        self._descriptions[key] = description
        if spec is not None:
            self._specs[key] = spec
        logger.info("Registered external tool: %s", key)

    def get_spec(self, name: Optional[str]) -> Optional["ApiToolSpec"]:
        return self._specs.get((name or "").strip().lower())

    def has(self, name: Optional[str]) -> bool:
        return bool(name) and name.strip().lower() in self._tools

    def names(self) -> List[str]:
        return sorted(self._tools)

    def descriptions(self) -> Dict[str, str]:
        """Return {tool_name: description} for all registered tools."""
        return dict(self._descriptions)

    def describe(self) -> str:
        """Human/LLM-readable catalog of available tools."""
        if not self._descriptions:
            return "(no external tools registered)"
        return "\n".join(f"- {n}: {d}" for n, d in sorted(self._descriptions.items()))

    def call(self, name: str, parameters: Dict[str, Any]) -> ExternalToolResult:
        key = (name or "").strip().lower()
        tool = self._tools.get(key)
        if tool is None:
            return ExternalToolResult(
                ok=False, tool=key, error=f"Unknown external tool '{name}'."
            )
        try:
            result = tool(parameters or {})
            result.tool = result.tool or key
            return result
        except Exception as exc:  # never let a tool crash the request
            logger.error("External tool '%s' raised: %s", key, exc, exc_info=True)
            return ExternalToolResult(ok=False, tool=key, error=str(exc))


# ---------------------------------------------------------------------------
# Default tools (safe stubs). Replace / extend with real integrations.
# ---------------------------------------------------------------------------

def _echo_tool(parameters: Dict[str, Any]) -> ExternalToolResult:
    """Placeholder tool. Returns the parameters it received.

    Useful as a safe default so the routing + fallback flow works end-to-end
    before real external APIs are wired in.
    """
    return ExternalToolResult(
        ok=True,
        tool="echo",
        text=(
            "External API placeholder response. No live integration is "
            "configured yet. Received parameters: "
            + json.dumps(parameters, default=str)
        ),
        data=parameters,
    )


def build_default_registry() -> ExternalAPIRegistry:
    """Registry with safe placeholder tools. Extend in production."""
    registry = ExternalAPIRegistry()
    registry.register(
        "echo",
        _echo_tool,
        "Generic placeholder for live/external data. Use when a query needs "
        "real-time information not stored in the knowledge base.",
    )
    return registry


# ---------------------------------------------------------------------------
# User-supplied HTTP APIs (passed per request from the UI)
# ---------------------------------------------------------------------------

@dataclass
class ApiToolSpec:
    """Declarative spec for an external tool supplied by the user.

    `kind` selects the connector that executes the tool, mirroring how the
    retrieval layer selects a vector-store backend from `vector_store_id`:

      - "http" (default): a REST/HTTP API. Uses url/method/headers/
        query_params/json_body.
      - "database": a SQL/NoSQL database query, executed by reusing the
        existing services.sources.database_source.DatabaseSource.
      - "api": an external JSON API fetched/paginated by reusing
        services.sources.api_source.APISource.
      - "cloud_storage": cloud bucket listing/fetch via
        services.sources.cloud_storage_source.CloudStorageSource.

    For non-http kinds, `config` carries the service's `source_details`
    (credentials, host, query, bucket, ...) — the same shape those existing
    source services already accept. `config` values also support
    `{placeholder}` templating filled from the query.

    Templating: any string value in `url`, `query_params`, `headers`,
    `json_body`, or `config` may contain `{placeholders}` filled from the
    router's extracted parameters (always includes `{query}`). Unknown
    placeholders are left blank.
    """
    name: str
    description: str
    url: str = ""
    method: str = "GET"
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    json_body: Optional[Dict[str, Any]] = None
    timeout: int = 20
    # Optional dotted path into the JSON response to extract the useful text,
    # e.g. "current.condition.text". If unset, the whole JSON is returned.
    response_path: Optional[str] = None
    # Connector selector + generic per-service config for non-http kinds.
    kind: str = "http"
    config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ApiToolSpec":
        name = str(d.get("name") or d.get("tool") or "").strip()
        url = str(d.get("url") or "").strip()

        # Resolve the connector kind. Explicit `kind`/`type` wins; otherwise
        # infer from the shape (db_credentials/source_details -> database, a
        # bare url -> http). This keeps old url-only payloads working as http.
        kind = str(d.get("kind") or d.get("type") or "").strip().lower()
        config = dict(d.get("config") or d.get("source_details") or {})
        # A db_credentials block always contributes connection fields to the
        # config (merged in, without overwriting explicit config values) and
        # signals a database tool.
        db_creds = dict(d.get("db_credentials") or {})
        has_db_creds = bool(db_creds)
        if has_db_creds:
            for k, v in db_creds.items():
                config.setdefault(k, v)
        if not kind:
            if has_db_creds or config.get("db_type") or config.get("database_type_name") \
                    or config.get("host") or config.get("connection_uri"):
                kind = "database"
            elif config.get("provider") or config.get("bucket"):
                kind = "cloud_storage"
            else:
                kind = "http"

        if not name:
            raise ValueError("Each tool needs a 'name'.")
        if kind == "http" and not url:
            raise ValueError("An http tool needs a 'url'.")
        if kind != "http" and not config:
            raise ValueError(f"A '{kind}' tool needs a 'config' (service credentials).")

        return cls(
            name=name,
            description=str(d.get("description") or f"External tool '{name}'."),
            url=url,
            kind=kind,
            config=config,
            method=str(d.get("method") or "GET").upper(),
            headers=dict(d.get("headers") or {}),
            query_params=dict(d.get("query_params") or d.get("params") or {}),
            json_body=d.get("json_body") if isinstance(d.get("json_body"), dict) else None,
            timeout=int(d.get("timeout") or 20),
            response_path=d.get("response_path"),
        )


def _fill_template(value: Any, params: Dict[str, Any]) -> Any:
    """Recursively substitute {placeholders} in strings using params.

    Whole-value placeholders preserve type: if a string is EXACTLY "{key}"
    (nothing before/after), the substituted value keeps its real type from
    params — so "{isActive}" can become a JSON boolean true, and "{price}" a
    JSON number, not the strings "true"/"1999". This matters for APIs whose
    body fields are typed (bool/int). Mixed strings like "page {n}" still do
    normal string interpolation.
    """
    if isinstance(value, str):
        whole = re.fullmatch(r"\{(\w+)\}", value.strip())
        if whole:
            key = whole.group(1)
            if key in params:
                return params[key]  # keep the real type (bool/int/float/str)
            return ""  # unknown whole-value placeholder -> blank
        def repl(m):
            key = m.group(1)
            return str(params.get(key, ""))
        return re.sub(r"\{(\w+)\}", repl, value)
    if isinstance(value, dict):
        return {k: _fill_template(v, params) for k, v in value.items()}
    if isinstance(value, list):
        return [_fill_template(v, params) for v in value]
    return value


def _dig(data: Any, path: Optional[str]) -> Any:
    """Follow a dotted path into nested dict/list data; return data if no path."""
    if not path:
        return data
    cur = data
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list) and part.isdigit():
            idx = int(part)
            cur = cur[idx] if 0 <= idx < len(cur) else None
        else:
            return None
    return cur


def make_http_tool(spec: ApiToolSpec) -> ExternalTool:
    """Build an ExternalTool callable that invokes a user-supplied HTTP API."""
    import json as _json

    import requests  # local import keeps query_router importable without requests

    def _tool(parameters: Dict[str, Any]) -> ExternalToolResult:
        try:
            url = _fill_template(spec.url, parameters)
            headers = _fill_template(spec.headers, parameters)
            qp = _fill_template(spec.query_params, parameters)
            body = _fill_template(spec.json_body, parameters) if spec.json_body else None

            resp = requests.request(
                spec.method,
                url,
                headers=headers or None,
                params=qp or None,
                json=body,
                timeout=spec.timeout,
            )
            if resp.status_code >= 400:
                return ExternalToolResult(
                    ok=False, tool=spec.name,
                    error=f"API '{spec.name}' returned HTTP {resp.status_code}: {resp.text[:200]}",
                )
            try:
                data = resp.json()
            except ValueError:
                return ExternalToolResult(ok=True, tool=spec.name, text=resp.text[:4000], data=resp.text)

            extracted = _dig(data, spec.response_path)
            if isinstance(extracted, (dict, list)):
                text = _json.dumps(extracted, default=str)[:4000]
            else:
                text = str(extracted)
            return ExternalToolResult(ok=True, tool=spec.name, text=text, data=data)
        except Exception as exc:
            return ExternalToolResult(ok=False, tool=spec.name, error=str(exc))

    return _tool


# ---------------------------------------------------------------------------
# Service-backed connectors (reuse existing services.sources.* source classes)
# ---------------------------------------------------------------------------
# These mirror how the retrieval layer selects a backend from a request field:
# here the tool's `kind` selects which existing source service runs it. No new
# connection/query logic is written — we delegate to the mature DatabaseSource /
# APISource / CloudStorageSource classes, passing the templated `config` as
# their `source_details`.

def _run_source_service(spec: "ApiToolSpec", parameters: Dict[str, Any]) -> ExternalToolResult:
    """Execute a non-http tool by delegating to the matching source service.

    `spec.config` (templated with query params) becomes the service's
    `source_details`. The service's async `discover()` returns records/rows,
    which we normalize into an ExternalToolResult.
    """
    import asyncio
    import json as _json

    # Fill {placeholders} in the config (e.g. a query or collection name from
    # the user's natural-language request).
    config = _fill_template(dict(spec.config or {}), parameters)
    request_data = {"source_details": config, "source_type_name": spec.kind}

    kind = (spec.kind or "").lower()
    try:
        if kind == "database":
            try:
                from retrieval.sources.database_source import DatabaseSource
            except ImportError:
                from services.sources.database_source import DatabaseSource
            source = DatabaseSource(request_data)
        elif kind == "api":
            try:
                from retrieval.sources.api_source import APISource
            except ImportError:
                from services.sources.api_source import APISource
            source = APISource(request_data)
        elif kind in ("cloud_storage", "cloud", "storage"):
            try:
                from retrieval.sources.cloud_storage_source import CloudStorageSource
            except ImportError:
                from services.sources.cloud_storage_source import CloudStorageSource
            provider = str(config.get("provider") or "").lower()
            source = CloudStorageSource(request_data, provider)
        else:
            return ExternalToolResult(
                ok=False, tool=spec.name,
                error=f"Unsupported tool kind '{spec.kind}'.",
            )

        # The source services are async; run discover() to completion. Detect
        # whether we're already inside a running event loop (e.g. a FastAPI
        # request handler). If so, run the coroutine in a fresh loop on a
        # worker thread to avoid 'loop already running'; otherwise run it
        # directly. asyncio.get_event_loop() is deprecated on 3.12+ and raises
        # when there is no current loop, so we probe with get_running_loop().
        try:
            asyncio.get_running_loop()
            inside_loop = True
        except RuntimeError:
            inside_loop = False

        if inside_loop:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                records = ex.submit(lambda: asyncio.run(source.discover())).result(
                    timeout=spec.timeout
                )
        else:
            records = asyncio.run(source.discover())

        text = _json.dumps(records, default=str)[:4000]
        return ExternalToolResult(
            ok=True, tool=spec.name, text=text, data=records,
        )
    except Exception as exc:
        return ExternalToolResult(ok=False, tool=spec.name, error=str(exc))


def make_tool(spec: "ApiToolSpec") -> ExternalTool:
    """Build the executable for a tool, dispatching on `spec.kind`.

    http -> make_http_tool (unchanged). Other kinds -> the matching existing
    source service (database/api/cloud_storage). This is the single generic
    extension point: add a kind here and the whole router/extraction pipeline
    works with it unchanged.
    """
    if (spec.kind or "http").lower() == "http":
        return make_http_tool(spec)

    def _tool(parameters: Dict[str, Any]) -> ExternalToolResult:
        return _run_source_service(spec, parameters)

    return _tool


def collect_placeholders(spec: "ApiToolSpec") -> List[str]:
    """Return the set of {placeholder} names used across a tool spec.

    Scans url, headers, query_params, and json_body for `{name}` tokens so the
    extractor knows exactly which fields it must pull from the user query.
    `query`/`q` are always available (the raw query) so they're excluded here.
    """
    names: set = set()

    def _scan(value: Any) -> None:
        if isinstance(value, str):
            for m in re.finditer(r"\{(\w+)\}", value):
                names.add(m.group(1))
        elif isinstance(value, dict):
            for v in value.values():
                _scan(v)
        elif isinstance(value, list):
            for v in value:
                _scan(v)

    _scan(spec.url)
    _scan(spec.headers)
    _scan(spec.query_params)
    _scan(spec.json_body)
    _scan(spec.config)          # non-http tools template their config too
    names.discard("query")
    names.discard("q")
    return sorted(names)


def _coerce_scalar(value: Any) -> Any:
    """Coerce a string value to int/float/bool when it clearly is one.

    The LLM returns everything as strings. APIs often need real numbers/bools
    (e.g. price 1999, available true). Convert safely; leave plain text as-is.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if s == "":
        return value
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    # int (no decimal point, optional sign)
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except ValueError:
            return value
    # float
    if re.fullmatch(r"[+-]?\d*\.\d+", s):
        try:
            return float(s)
        except ValueError:
            return value
    return value


# Placeholder names that denote a record identifier for update/delete targets.
_ID_PLACEHOLDER_HINTS = ("id", "object_id", "objectid", "book_id", "record_id", "uuid")

# Matches id-like tokens in a query: UUIDs, long alnum ids, letter+digit codes.
_ID_TOKEN_RE = re.compile(
    r"\b("
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"  # UUID
    r"|[A-Za-z]{1,}-?\d{2,}[\w-]*"   # bk-013, EMP0001, obj_99
    r"|[0-9a-fA-F]{16,}"             # long hex id
    r"|\d{3,}"                        # bare numeric id
    r")\b"
)


def _guess_identifier_from_query(query: str) -> Optional[str]:
    """Return the most id-like token in the query, if any (longest wins)."""
    if not query:
        return None
    candidates = _ID_TOKEN_RE.findall(query)
    if not candidates:
        return None
    # Prefer a UUID, else the longest token (most specific).
    for c in candidates:
        if len(c) >= 32 and "-" in c:
            return c
    return max(candidates, key=len)


def extract_tool_parameters(
    query: str,
    spec: "ApiToolSpec",
    *,
    generate_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
) -> Dict[str, Any]:
    """Extract values for a tool's {placeholders} from the user query.

    Sources, in order:
      1. Embedded JSON — if the query contains a JSON object, its keys (and one
         nested level) become parameters directly.
      2. Identifier heuristic — fills id-like placeholders (id/object_id/...)
         from an id token in the query (UUID, bk-013, EMP0001, long numbers),
         so update/delete targets resolve without the LLM.
      3. LLM extraction — pulls remaining placeholders from the natural-language
         query. Values are coerced to int/float/bool when appropriate.

    Always returns at least {"query": query, "q": query}. Never raises — on any
    failure it returns what it has so templating degrades gracefully.
    """
    params: Dict[str, Any] = {"query": query or "", "q": query or ""}

    # 1. Embedded JSON object in the query (e.g. user pasted the body).
    embedded = _extract_json(query or "")
    if isinstance(embedded, dict):
        for k, v in embedded.items():
            params.setdefault(k, v)
        # Also flatten one nested level (e.g. {"data": {"price": 1}} -> price)
        for v in embedded.values():
            if isinstance(v, dict):
                for nk, nv in v.items():
                    params.setdefault(nk, nv)

    placeholders = collect_placeholders(spec)

    # 2. Identifier heuristic for id-like placeholders (update/delete targets).
    id_token = _guess_identifier_from_query(query or "")
    if id_token:
        for p in placeholders:
            if p.lower() in _ID_PLACEHOLDER_HINTS and params.get(p) in (None, ""):
                params[p] = id_token

    missing = [p for p in placeholders if p not in params or params.get(p) in (None, "")]

    # 3. LLM extraction for whatever placeholders the query text still owes.
    if missing and generate_fn is not None:
        try:
            fields_list = ", ".join(missing)
            system = (
                "You extract field values from a user's instruction to call an API. "
                "Return STRICT JSON only (no prose): an object whose keys are exactly "
                "the requested field names. Rules: (1) Use the value exactly as the "
                "user stated it. (2) For numbers return a JSON number. (3) For "
                "boolean-looking fields return a JSON boolean true/false — map "
                "active/enable/enabled/on/yes/visible to true, and "
                "inactive/disable/disabled/off/no/hidden to false. (4) If a field's "
                "value is genuinely not in the text, set it to an empty string. "
                "(5) Never invent or guess values. (6) Resolve relative dates and times "
                "('tomorrow', 'next Monday', 'in 2 hours') to concrete values using "
                f"today's date. {_today_line()}"
            )
            user = (
                f"User instruction: {query}\n\n"
                f"Extract these fields: {fields_list}\n"
                "Return a JSON object with exactly those keys."
            )
            raw = generate_fn([
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ])
            parsed = _extract_json(raw)
            if isinstance(parsed, dict):
                for k, v in parsed.items():
                    if k in missing and v not in (None, ""):
                        params[k] = _coerce_scalar(v)
        except Exception as exc:
            logger.warning("Tool parameter extraction failed: %s", exc)

    return params


def registry_from_api_specs(specs: List[Dict[str, Any]]) -> Tuple[ExternalAPIRegistry, List[str]]:
    """Build a registry from user-supplied API specs (per request).

    Returns (registry, warnings). Invalid specs are skipped with a warning so
    one bad entry never breaks the whole request.
    """
    registry = ExternalAPIRegistry()
    warnings: List[str] = []
    for raw in specs or []:
        try:
            spec = ApiToolSpec.from_dict(raw)
            registry.register(spec.name, make_tool(spec), spec.description, spec=spec)
        except Exception as exc:
            warnings.append(f"Skipped an API spec: {exc}")
    return registry, warnings


# ===========================================================================
# Heuristic router (fallback)
# ===========================================================================

# Terms that strongly imply live / dynamic / external data needs.
_LIVE_DATA_PATTERNS = [
    r"\bweather\b", r"\bforecast\b", r"\btemperature\b",
    r"\bstock price\b", r"\bshare price\b", r"\bexchange rate\b",
    r"\bcurrency\b", r"\bcrypto\b", r"\bbitcoin\b",
    r"\bnews\b", r"\bheadlines\b", r"\blatest\b", r"\brecent(ly)?\b",
    r"\bcurrent(ly)?\b", r"\btoday\b", r"\btonight\b", r"\byesterday\b",
    r"\bright now\b", r"\bnow\b", r"\bthis (week|month|year)\b",
    r"\blive\b", r"\breal[- ]?time\b", r"\bscore\b", r"\bschedule\b",
    r"\btraffic\b", r"\bflight status\b",
]

# Terms that imply the answer lives in ingested documents / the KB.
_KB_PATTERNS = [
    r"\bdocument\b", r"\bmy (doc|file|pdf|notes?)\b", r"\baccording to\b",
    r"\bin the (doc|file|report|policy|manual)\b", r"\bsummar(y|ize|ise)\b",
    r"\bknowledge base\b", r"\bingested\b", r"\buploaded\b",
    r"\bsection\b", r"\bchapter\b", r"\bpolicy\b", r"\bexplain\b",
]

_LIVE_RE = re.compile("|".join(_LIVE_DATA_PATTERNS), re.IGNORECASE)
_KB_RE = re.compile("|".join(_KB_PATTERNS), re.IGNORECASE)

_ACTION_PATTERNS = [
    r"\bactivate\b", r"\bdeactivate\b", r"\benable\b", r"\bdisable\b",
    r"\bturn (on|off)\b", r"\bset\b", r"\btoggle\b", r"\bswitch\b",
    r"\bcreate\b", r"\badd\b", r"\binsert\b", r"\bregister\b",
    r"\bupdate\b", r"\bedit\b", r"\bmodify\b", r"\bchange\b", r"\bpatch\b",
    r"\bdelete\b", r"\bremove\b", r"\bdrop\b", r"\bdestroy\b",
    r"\bstart\b", r"\bstop\b", r"\brestart\b", r"\bpause\b", r"\bresume\b",
    r"\brename\b", r"\bmove\b", r"\bcopy\b", r"\bupload\b", r"\btrigger\b",
    r"\brun\b", r"\bexecute\b", r"\bperform\b", r"\binvoke\b", r"\bcall\b",
]
_ACTION_RE = re.compile("|".join(_ACTION_PATTERNS), re.IGNORECASE)

# Words with no selection signal — ignored when scoring a query against a
# tool's name/description so common verbs/articles don't inflate matches.
_SELECT_STOPWORDS = {
    "the", "a", "an", "to", "of", "for", "in", "on", "and", "or", "with", "by",
    "is", "are", "be", "this", "that", "it", "its", "as", "at", "from", "into",
    "please", "me", "my", "status", "set", "get", "all", "any", "via", "using",
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set:
    """Lowercase word tokens of a string, minus selection stopwords."""
    return {
        w for w in _WORD_RE.findall((text or "").lower())
        if w not in _SELECT_STOPWORDS and len(w) > 1
    }


def select_best_tool(query: str, registry: "ExternalAPIRegistry") -> Tuple[Optional[str], float]:
    """Pick the tool whose name+description best matches the query.

    Scores each tool by token overlap between the query and the tool's
    name + description (name tokens weighted higher). Returns
    (tool_name, score) where score is a 0..1 confidence. Falls back to the
    first registered tool with a low score when nothing overlaps, so a tool is
    still chosen when the caller forces an API route. API-agnostic: no tool,
    service, or verb is hardcoded into the match.
    """
    names = registry.names()
    if not names:
        return None, 0.0

    q_tokens = _tokenize(query)
    if not q_tokens:
        return names[0], 0.0

    descriptions = registry.descriptions()
    best_name: Optional[str] = None
    best_score = -1.0

    for name in names:
        name_tokens = _tokenize(name.replace("_", " "))
        desc_tokens = _tokenize(descriptions.get(name, ""))

        # Name matches are worth more than description matches.
        name_overlap = len(q_tokens & name_tokens)
        desc_overlap = len(q_tokens & desc_tokens)
        raw = 2.0 * name_overlap + 1.0 * desc_overlap

        # Normalize against how many query tokens could possibly match, so a
        # short precise query and a long one are comparable.
        denom = max(len(q_tokens), 1)
        score = min(raw / denom, 1.0)

        if score > best_score:
            best_score = score
            best_name = name

    # Nothing overlapped at all — fall back to the first tool, low confidence.
    if best_score <= 0:
        return names[0], 0.0
    return best_name, best_score


def heuristic_route(query: str, registry: ExternalAPIRegistry) -> RouteDecision:
    """Rule-based routing used when the LLM router is unavailable.

    Routing signals (in priority order):
      1. ACTION/CRUD verb (activate/create/delete/...) + a registered tool ->
         API, with the best-matching tool selected by name/description overlap.
      2. Live-data cue (weather/news/now/...) -> API (or HYBRID with a KB cue).
      3. Otherwise -> RAG (the knowledge base).
    """
    q = query or ""
    live_hits = len(_LIVE_RE.findall(q))
    kb_hits = len(_KB_RE.findall(q))
    action_hits = len(_ACTION_RE.findall(q))
    has_tools = bool(registry.names())

    best_tool, match_score = select_best_tool(q, registry) if has_tools else (None, 0.0)

    # 1. Action/CRUD intent + a tool available -> perform the operation.
    if action_hits > 0 and has_tools:
        return RouteDecision(
            source=RouteSource.API,
            tool=best_tool,
            parameters={"query": q},
            reason=(
                f"Matched {action_hits} action verb(s); selected tool "
                f"'{best_tool}' (match {match_score:.2f})."
            ),
            # Confidence blends the action signal with how well a tool matched.
            confidence=min(0.6 + 0.1 * action_hits + 0.2 * match_score, 0.95),
            decided_by="heuristic",
        )

    # 2. Live-data intent.
    if live_hits > 0 and kb_hits == 0 and has_tools:
        return RouteDecision(
            source=RouteSource.API,
            tool=best_tool,
            parameters={"query": q},
            reason=f"Matched {live_hits} live-data cue(s); selected tool '{best_tool}'.",
            confidence=min(0.5 + 0.1 * live_hits + 0.2 * match_score, 0.9),
            decided_by="heuristic",
        )
    if live_hits > 0 and kb_hits > 0 and has_tools:
        return RouteDecision(
            source=RouteSource.HYBRID,
            tool=best_tool,
            parameters={"query": q},
            reason="Matched both live-data and knowledge-base cues.",
            confidence=0.6,
            decided_by="heuristic",
        )

    # 3. Default: knowledge base.
    return RouteDecision(
        source=RouteSource.RAG,
        reason="No action/live-data cue detected; defaulting to the knowledge base.",
        confidence=0.6 if kb_hits else 0.55,
        decided_by="heuristic",
    )


# ===========================================================================
# LLM router (primary)
# ===========================================================================

_ROUTER_SYSTEM_PROMPT = (
    "You are a routing classifier for a Knowledge Base (RAG) assistant. Decide "
    "where a user query should be answered from. Respond with STRICT JSON only, "
    "no prose.\n\n"
    "DEFAULT TO \"rag\". The knowledge base holds ingested documents and domain "
    "facts (profiles, statistics, histories, summaries, policies). Most factual, "
    "historical, biographical, or 'compare/explain/how many' questions are "
    "answered from it — including sports stats, company data, and records that "
    "are stored in documents.\n\n"
    "Only choose \"api\" when BOTH are true:\n"
    "  (1) the question clearly needs volatile, real-time data (today's weather, "
    "a live market price, right-now status), AND\n"
    "  (2) one of the AVAILABLE TOOLS below actually serves that data.\n"
    "If no available tool matches the query's topic, you MUST use \"rag\" (never "
    "force an unrelated tool). For example, do not route a cricket-statistics "
    "question to a weather tool — use \"rag\".\n\n"
    "Use \"hybrid\" only when the answer genuinely needs BOTH stored documents "
    "and a matching live tool.\n\n"
    "AVAILABLE EXTERNAL TOOLS:\n{tools}\n\n"
    "Guidance: 'statistics', 'records', 'career', 'history', 'as of <year>' are "
    "usually STORED facts -> rag, not live data. Treat data as live only when it "
    "changes minute-to-minute and a tool provides it.\n\n"
    "Return JSON with keys: source (rag|api|hybrid), tool (an available tool name "
    "whose topic matches the query, or null), parameters (object; include a "
    "\"query\" string and any extracted args like location/symbol/date), reason "
    "(short string), confidence (0..1)."
)


def _today_line() -> str:
    """Today's date for prompts, so relative dates ('tomorrow') resolve correctly."""
    return f"Today's date is {datetime.now():%Y-%m-%d (%A)}."


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of an LLM response."""
    if not text:
        return None
    # Strip code fences if present.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else None
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def llm_route(
    query: str,
    registry: ExternalAPIRegistry,
    *,
    generate_fn: Callable[[List[Dict[str, str]]], str],
) -> Optional[RouteDecision]:
    """Classify the query with the LLM. Returns None on any failure.

    `generate_fn` is injected (the existing `_generate_answer_gpt`) so this
    module stays decoupled from the OpenAI config in retrieval_new.py.
    """
    try:
        system = _ROUTER_SYSTEM_PROMPT.format(tools=registry.describe()) + " " + _today_line()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Query: {query}"},
        ]
        raw = generate_fn(messages)
        parsed = _extract_json(raw)
        if not parsed:
            logger.warning("LLM router returned unparseable output: %r", (raw or "")[:200])
            return None

        source_raw = str(parsed.get("source", "")).strip().lower()
        if source_raw not in {"rag", "api", "hybrid"}:
            logger.warning("LLM router returned invalid source: %r", source_raw)
            return None

        source = RouteSource(source_raw)
        tool = parsed.get("tool")
        if isinstance(tool, str):
            tool = tool.strip() or None
        else:
            tool = None

        # If the model picked a tool we don't have, choose the best-matching
        # one by name/description instead of blindly taking the first.
        if source in (RouteSource.API, RouteSource.HYBRID):
            if not registry.has(tool):
                best, _ = select_best_tool(query, registry)
                tool = best

        params = parsed.get("parameters")
        if not isinstance(params, dict):
            params = {}
        params.setdefault("query", query)

        try:
            confidence = float(parsed.get("confidence", 0.7))
        except (TypeError, ValueError):
            confidence = 0.7

        return RouteDecision(
            source=source,
            tool=tool,
            parameters=params,
            reason=str(parsed.get("reason", "")).strip(),
            confidence=max(0.0, min(confidence, 1.0)),
            decided_by="llm",
        )
    except Exception as exc:
        logger.error("LLM router failed: %s", exc, exc_info=True)
        return None


def decide_route(
    query: str,
    registry: ExternalAPIRegistry,
    *,
    generate_fn: Optional[Callable[[List[Dict[str, str]]], str]] = None,
    use_llm: bool = True,
) -> RouteDecision:
    """Decide the route: try the LLM router, fall back to heuristics.

    If `source == api` but no external tools are registered, the decision is
    downgraded to `rag` so the request can still be answered.
    """
    decision: Optional[RouteDecision] = None
    if use_llm and generate_fn is not None:
        decision = llm_route(query, registry, generate_fn=generate_fn)

    if decision is None:
        decision = heuristic_route(query, registry)

    # Safety net: can't route to API without any tool.
    if decision.source in (RouteSource.API, RouteSource.HYBRID) and not registry.has(decision.tool):
        if registry.names():
            best, _ = select_best_tool(query, registry)
            decision.tool = best
        else:
            logger.warning(
                "Route '%s' requested but no external tools registered — "
                "falling back to RAG.", decision.source.value,
            )
            decision.source = RouteSource.RAG
            decision.tool = None
            decision.reason = (decision.reason + " | ").strip() + \
                "No external tools available; fell back to RAG."
    return decision
