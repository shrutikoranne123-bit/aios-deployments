import copy
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx

try:
    from retrieval._vendor.api_models import (
        APISourceInput,
        AuthType,
        HttpMethod,
        PaginationType,
        PostPaginationStrategy,
        WRITE_METHODS,
    )
except ImportError:
    from services.api_upload import (
        APISourceInput,
        AuthType,
        HttpMethod,
        PaginationType,
        PostPaginationStrategy,
        WRITE_METHODS,
    )
try:
    from retrieval._vendor.json_extractor import JSONExtractor
except ImportError:
    from services.extraction.extractors.json_extractor import JSONExtractor
try:
    from retrieval._vendor.tokenizer_service import get_max_tokens, get_tokenizer
except ImportError:
    from services.tokenizer_service import get_max_tokens, get_tokenizer

logger = logging.getLogger(__name__)


class APISourceService:
    """Fetch JSON from an external API and prepare it for downstream ingestion.

    Supports GET and all write methods (POST, PUT, PATCH).

    Pagination for write methods
    ────────────────────────────
    Two strategies are available, configured via
    ``PaginationConfig.post_pagination_strategy``:

    * ``PostPaginationStrategy.QUERY`` (default) — pagination params are
      appended to the query-string exactly as they are for GET requests.
      The original request body is sent unchanged on every page.

    * ``PostPaginationStrategy.BODY`` — pagination params are merged *into*
      the JSON request body on each page.  The original body is deep-copied
      and updated so that the base payload (filters, search terms, etc.)
      is preserved across pages.
    """

    DEFAULT_TIMEOUT_SECONDS = 30.0
    MAX_PAGES = 50

    async def fetch_and_extract(
        self,
        payload: APISourceInput,
        embedding_details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Fetch paginated API data and prepare both JSON and flattened outputs."""
        headers, auth = await self._build_auth(payload)
        extractor = JSONExtractor({
            "fmt": "json",
            "json_path": payload.json_path,
            "field_paths": payload.field_paths,
        })
        tokenizer, safe_token_limit = self._build_token_budget(embedding_details or {})

        aggregated_entries: List[Any] = []
        flat_rows: List[Dict[str, Any]] = []
        page_urls: List[str] = []
        pages_fetched = 0
        record_counter = 0
        page_marker = payload.pagination.start_page or 1

        is_write_method = payload.http_method in WRITE_METHODS
        client_kwargs = self._build_client_kwargs(payload)

        async with httpx.AsyncClient(
            timeout=self.DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
            **client_kwargs,
        ) as client:
            while pages_fetched < self.MAX_PAGES:
                # Build query-string params and (for write methods) the body
                # for this specific page.
                query_params, request_body = self._build_request_params(
                    payload=payload,
                    page_marker=page_marker,
                    is_write_method=is_write_method,
                )

                logger.info(
                    "→ Fetching page %s | method=%s | params=%s | body_keys=%s",
                    page_marker,
                    payload.http_method.value,
                    query_params,
                    list(request_body.keys()) if isinstance(request_body, dict) else type(request_body).__name__,
                )

                response = await client.request(
                    method=payload.http_method.value,
                    url=str(payload.url),
                    headers=headers,
                    params=query_params if query_params else None,
                    # Only send a JSON body for write methods; None for GET/DELETE
                    json=request_body if is_write_method else None,
                    auth=auth,
                )
                response.raise_for_status()

                response_data = response.json()

                result = extractor.extract_data_result(
                    response_data,
                    source_name=payload.source_name,
                    base_metadata={"page_index": pages_fetched + 1},
                )

                if not result.success:
                    logger.warning(
                        "Extraction failed on page %s: %s", page_marker, result.error
                    )
                    break

                page_entries = self._attach_source_url(
                    [entry.raw_object for entry in result.entries],
                    str(response.url),
                )

                if not page_entries:
                    logger.info("No records on page %s. Stopping.", page_marker)
                    break

                aggregated_entries.extend(page_entries)
                page_urls.append(str(response.url))
                pages_fetched += 1

                for entry, raw_object in zip(result.entries, page_entries):
                    record_counter += 1
                    flat_rows.extend(
                        self._build_flat_rows(
                            entry=entry,
                            raw_object=raw_object,
                            tokenizer=tokenizer,
                            token_limit=safe_token_limit,
                            record_index=record_counter,
                            page_index=pages_fetched,
                            page_url=str(response.url),
                            api_url=str(payload.url),
                            source_name=payload.source_name,
                        )
                    )

                logger.info(
                    "Page %s returned %s records (Total: %s)",
                    page_marker,
                    len(page_entries),
                    len(aggregated_entries),
                )

                next_marker = self._resolve_next_page_marker(
                    payload=payload,
                    current_marker=page_marker,
                    page_size=payload.pagination.page_size_value or 20,
                    response_data=response_data,
                    page_entries_count=len(page_entries),
                )
                if next_marker is None:
                    logger.info("Last page detected. Stopping pagination.")
                    break

                page_marker = next_marker

        if not aggregated_entries:
            raise ValueError("No entries extracted from API")

        json_text = json.dumps(aggregated_entries, indent=2, ensure_ascii=False)

        logger.info(
            "✅ Pagination finished - %s pages, %s total records, %s flat row(s)",
            pages_fetched,
            len(aggregated_entries),
            len(flat_rows),
        )

        # Calculate content size from JSON response
        content_size_bytes = len(json_text.encode('utf-8'))

        return {
            "source_name": payload.source_name,
            "url": str(payload.url),
            "pages_fetched": pages_fetched,
            "entries_extracted": len(aggregated_entries),
            "flat_rows": flat_rows,
            "data": aggregated_entries,
            "markdown_text": json_text,
            "json_chunk_text": json_text,
            "page_urls": page_urls,
            "content_size_bytes": content_size_bytes,
        }

    # ──────────────────────────────────────────────────────────────────────────
    #  Request construction
    # ──────────────────────────────────────────────────────────────────────────

    def _build_request_params(
        self,
        payload: APISourceInput,
        page_marker: int,
        is_write_method: bool,
    ) -> Tuple[Dict[str, Any], Any]:
        """
        Return (query_params, request_body) for the current page.

        For GET / DELETE
        ─────────────────
        * query_params = user query_params + pagination params (always query-string)
        * request_body = None  (enforced by the Pydantic validator)

        For POST / PUT / PATCH — PostPaginationStrategy.QUERY (default)
        ────────────────────────────────────────────────────────────────
        * query_params = user query_params + pagination params
        * request_body = original body unchanged on every page

        For POST / PUT / PATCH — PostPaginationStrategy.BODY
        ──────────────────────────────────────────────────────
        * query_params = user query_params only (no pagination params)
        * request_body = deep copy of original body with pagination params merged in
        """
        pagination = payload.pagination
        strategy = pagination.post_pagination_strategy

        # Always start with the user-defined query params
        base_query_params: Dict[str, Any] = dict(payload.query_params)

        pagination_params = self._build_pagination_params(payload, page_marker)

        if not is_write_method:
            # GET / DELETE — all params go to the query-string
            return {**base_query_params, **pagination_params}, None

        # Write methods ───────────────────────────────────────────────────────
        if strategy == PostPaginationStrategy.QUERY or not pagination_params:
            # Pagination via query-string (or no pagination at all)
            query_params = {**base_query_params, **pagination_params}
            request_body = payload.request_body
            return query_params, request_body

        # PostPaginationStrategy.BODY — merge pagination params into the body
        base_body = payload.request_body
        if base_body is None:
            merged_body = dict(pagination_params)
        elif isinstance(base_body, dict):
            merged_body, merged_at_json_path = self._merge_into_request_body(
                request_body=base_body,
                json_path=payload.json_path,
                values=pagination_params,
            )
            if merged_at_json_path:
                logger.info(
                    "POST body pagination merged into request json_path '%s'",
                    payload.json_path,
                )
            else:
                # Deep-copy so the original APISourceInput is never mutated
                merged_body = {**copy.deepcopy(base_body), **pagination_params}
        else:
            # Non-dict body (e.g. a raw string/list): fall back to query-string
            logger.warning(
                "post_pagination_strategy='body' requested but request_body is %s, "
                "not a dict. Falling back to query-string pagination.",
                type(base_body).__name__,
            )
            return {**base_query_params, **pagination_params}, base_body

        return base_query_params, merged_body

    def _build_pagination_params(
        self,
        payload: APISourceInput,
        page_marker: int,
    ) -> Dict[str, Any]:
        """Return the pagination key-value pairs for the current page.

        Returns an empty dict when pagination_type is NONE.
        """
        if payload.pagination.pagination_type == PaginationType.NONE:
            return {}

        page_param = payload.pagination.page_param_name
        size_param = payload.pagination.page_size_param_name
        page_size  = payload.pagination.page_size_value

        if not page_param or not size_param or page_size is None:
            raise ValueError("Pagination parameters are incomplete")

        return {
            size_param: page_size,
            page_param: page_marker,
        }

    def _merge_into_request_body(
        self,
        request_body: Any,
        json_path: Optional[str],
        values: Dict[str, Any],
    ) -> Tuple[Any, bool]:
        """Merge values into a nested request-body object selected by json_path."""
        if (
            not isinstance(request_body, (dict, list))
            or not values
            or not str(json_path or "").strip()
        ):
            return request_body, False

        cloned_body = copy.deepcopy(request_body)
        target_node, matched = self._resolve_json_path_reference(cloned_body, json_path)
        if not matched:
            return request_body, False
        if not isinstance(target_node, dict):
            logger.warning(
                "Request json_path '%s' matched %s, not an object. "
                "Falling back to top-level POST body merge.",
                json_path,
                type(target_node).__name__,
            )
            return request_body, False

        target_node.update(values)
        return cloned_body, True

    def _resolve_json_path_reference(
        self,
        data: Any,
        path: Optional[str],
    ) -> Tuple[Any, bool]:
        """Resolve a simple JSON path against a request body and return the node."""
        tokens = self._parse_json_path(path)
        current = data

        for token_type, token_value in tokens:
            if token_type == "key":
                if not isinstance(current, dict) or token_value not in current:
                    return None, False
                current = current[token_value]
                continue

            if token_type == "index":
                if not isinstance(current, list):
                    return None, False
                if not (-len(current) <= token_value < len(current)):
                    return None, False
                current = current[token_value]
                continue

            return None, False

        return current, True

    @staticmethod
    def _parse_json_path(path: Optional[str]) -> List[Tuple[str, Any]]:
        """Parse a minimal JSONPath expression into key/index tokens."""
        raw_path = str(path or "").strip()
        if not raw_path or raw_path == "$":
            return []

        if raw_path.startswith("$."):
            raw_path = raw_path[2:]
        elif raw_path.startswith("$"):
            raw_path = raw_path[1:]
        elif not raw_path.startswith("["):
            raw_path = "." + raw_path

        tokens: List[Tuple[str, Any]] = []
        index = 0

        while index < len(raw_path):
            char = raw_path[index]

            if char == ".":
                index += 1
                start = index
                while index < len(raw_path) and raw_path[index] not in ".[":
                    index += 1
                key = raw_path[start:index].strip()
                if key:
                    tokens.append(("key", key))
                continue

            if char == "[":
                end = raw_path.find("]", index)
                if end == -1:
                    raise ValueError(f"Invalid JSON path {path!r}: missing ']'")
                content = raw_path[index + 1:end].strip()
                if (
                    len(content) >= 2
                    and content[0] in {"'", '"'}
                    and content[-1] == content[0]
                ):
                    tokens.append(("key", content[1:-1]))
                else:
                    try:
                        tokens.append(("index", int(content)))
                    except ValueError:
                        tokens.append(("key", content))
                index = end + 1
                continue

            start = index
            while index < len(raw_path) and raw_path[index] not in ".[":
                index += 1
            key = raw_path[start:index].strip()
            if key:
                tokens.append(("key", key))

        return tokens

    # ──────────────────────────────────────────────────────────────────────────
    #  Authentication
    # ──────────────────────────────────────────────────────────────────────────

    async def _build_auth(
        self,
        payload: APISourceInput,
    ) -> Tuple[Dict[str, str], Optional[httpx.Auth]]:
        headers = {
            "Accept": "application/json",
            **payload.headers,
        }
        # Add Content-Type for write methods that carry a body
        if payload.http_method in WRITE_METHODS and payload.request_body is not None:
            headers.setdefault("Content-Type", "application/json")

        auth: Optional[httpx.Auth] = None

        if payload.auth.type == AuthType.BASIC:
            auth = httpx.BasicAuth(
                payload.auth.username,
                payload.auth.password.get_secret_value(),
            )
        elif payload.auth.type == AuthType.BEARER:
            headers["Authorization"] = (
                f"Bearer {payload.auth.token.get_secret_value()}"
            )
        elif payload.auth.type == AuthType.API_KEY:
            headers[payload.auth.header_name] = payload.auth.api_key.get_secret_value()
        elif payload.auth.type == AuthType.OAUTH2:
            headers["Authorization"] = (
                f"Bearer {await self._get_oauth2_token(payload)}"
            )

        return headers, auth

    async def _get_oauth2_token(self, payload: APISourceInput) -> str:
        auth = payload.auth
        if auth.type != AuthType.OAUTH2:
            raise ValueError("OAuth2 token requested for non-OAuth2 configuration")

        request_data = {"grant_type": "client_credentials"}
        if auth.scope:
            request_data["scope"] = auth.scope

        client_kwargs = self._build_client_kwargs(payload)
        async with httpx.AsyncClient(
            timeout=self.DEFAULT_TIMEOUT_SECONDS,
            follow_redirects=True,
            **client_kwargs,
        ) as client:
            response = await client.post(
                str(auth.token_url),
                data=request_data,
                auth=httpx.BasicAuth(
                    auth.client_id,
                    auth.client_secret.get_secret_value(),
                ),
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            token_payload = response.json()

        access_token = token_payload.get("access_token")
        if not access_token:
            raise ValueError("OAuth2 token response did not include access_token")
        return access_token

    def _build_client_kwargs(
        self,
        payload: APISourceInput,
    ) -> Dict[str, Any]:
        """Build shared httpx client kwargs for TLS verification behavior."""
        ca_bundle_path = (payload.ca_bundle_path or "").strip() or None

        if ca_bundle_path:
            normalized_path = os.path.expandvars(os.path.expanduser(ca_bundle_path))
            if not os.path.exists(normalized_path):
                raise ValueError(
                    f"Custom CA bundle path does not exist: {normalized_path}"
                )
            logger.info("API source using custom CA bundle: %s", normalized_path)
            return {"verify": normalized_path}

        if not payload.verify_ssl:
            logger.warning(
                "TLS certificate verification is disabled for API source '%s' (%s). "
                "Use this only for trusted internal endpoints.",
                payload.source_name,
                payload.url,
            )
            return {"verify": False}

        return {}

    # ──────────────────────────────────────────────────────────────────────────
    #  Pagination helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _next_page_marker(
        self,
        payload: APISourceInput,
        page_marker: int,
        page_size: int,
    ) -> int:
        if payload.pagination.pagination_type == PaginationType.PAGE_NUMBER:
            return page_marker + 1
        if payload.pagination.pagination_type == PaginationType.OFFSET:
            return page_marker + page_size
        return page_marker

    def _resolve_next_page_marker(
        self,
        payload: APISourceInput,
        current_marker: int,
        page_size: int,
        response_data: Any,
        page_entries_count: int,
    ) -> Optional[int]:
        pagination = (
            response_data.get("pagination", {}) if isinstance(response_data, dict) else {}
        )

        has_next = pagination.get("has_next")
        if has_next is False:
            return None

        next_page = self._safe_int(pagination.get("next_page"))
        if next_page is not None:
            return next_page

        next_offset = self._safe_int(pagination.get("next_offset"))
        if next_offset is not None:
            return next_offset

        total_pages  = self._safe_int(pagination.get("total_pages"))
        current_page = self._safe_int(pagination.get("page"))
        if total_pages is not None and current_page is not None:
            if current_page >= total_pages:
                return None
            return current_page + 1

        total_items = self._safe_int(pagination.get("total_items"))
        if (
            payload.pagination.pagination_type == PaginationType.OFFSET
            and total_items is not None
        ):
            next_marker = current_marker + page_size
            return next_marker if next_marker < total_items else None

        if page_entries_count < page_size:
            return None

        return self._next_page_marker(payload, current_marker, page_size)

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        try:
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    # ──────────────────────────────────────────────────────────────────────────
    #  Token budget
    # ──────────────────────────────────────────────────────────────────────────

    def _build_token_budget(
        self,
        embedding_details: Dict[str, Any],
    ) -> Tuple[Any, int]:
        model_name = str(
            embedding_details.get("embedding_model_name")
            or embedding_details.get("resolved_model_name")
            or "bge-base"
        ).lower().replace("_", "-").strip()

        tokenizer = None
        try:
            tokenizer = get_tokenizer(model_name)
        except Exception as exc:
            logger.warning("Could not load tokenizer for %s: %s", model_name, exc)

        max_tokens = get_max_tokens(model_name)
        safe_limit = max(64, max_tokens - self._resolve_token_margin(max_tokens))
        return tokenizer, safe_limit

    @staticmethod
    def _resolve_token_margin(max_tokens: int) -> int:
        if max_tokens <= 128:
            return 8
        if max_tokens <= 512:
            return 16
        return min(64, max(16, max_tokens // 32))

    # ──────────────────────────────────────────────────────────────────────────
    #  Entry assembly
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _attach_source_url(
        entries: List[Any],
        page_url: str,
    ) -> List[Any]:
        annotated_entries: List[Any] = []
        for entry in entries:
            if isinstance(entry, dict):
                annotated_entries.append({**entry, "_source_url": page_url})
            else:
                annotated_entries.append({"value": entry, "_source_url": page_url})
        return annotated_entries

    def _build_flat_rows(
        self,
        entry: Any,
        raw_object: Any,
        tokenizer: Any,
        token_limit: int,
        record_index: int,
        page_index: int,
        page_url: str,
        api_url: str,
        source_name: str,
    ) -> List[Dict[str, Any]]:
        searchable_object = self._strip_private_fields(raw_object)
        flat_text = self._serialize_flat_record(searchable_object)
        if not flat_text:
            return []

        entry_id = self._resolve_record_identifier(
            raw_object,
            getattr(entry, "entry_id", None),
            record_index,
        )
        display_name = (
            getattr(entry, "display_name", None)
            or self._resolve_display_name(raw_object, entry_id, record_index)
        )

        parts = self._split_flat_record_text(
            flat_text=flat_text,
            tokenizer=tokenizer,
            token_limit=token_limit,
        )

        rows: List[Dict[str, Any]] = []
        part_count = len(parts)
        for part_index, text in enumerate(parts, start=1):
            rows.append(
                {
                    "text": text,
                    "record_index": record_index,
                    "entry_id": entry_id,
                    "entry_display_name": display_name,
                    "page_index": page_index,
                    "page_url": page_url,
                    "api_url": api_url,
                    "source_name": source_name,
                    "part_index": part_index,
                    "part_count": part_count,
                }
            )

        return rows

    @staticmethod
    def _strip_private_fields(raw_object: Any) -> Any:
        if isinstance(raw_object, dict):
            return {
                key: value
                for key, value in raw_object.items()
                if not str(key).startswith("_")
            }
        return raw_object

    def _serialize_flat_record(self, raw_object: Any) -> str:
        if self._is_scalar(raw_object):
            return self._scalar_value(raw_object)
        return "\n".join(self._flatten_grouped_lines(raw_object)).strip()

    def _flatten_grouped_lines(
        self,
        value: Any,
        prefix: str = "",
    ) -> List[str]:
        if isinstance(value, dict):
            if prefix:
                grouped_values: List[str] = []
                nested_lines: List[str] = []
                for key, item in value.items():
                    child_key = str(key)
                    child_prefix = f"{prefix}.{child_key}"
                    if self._is_scalar(item):
                        grouped_values.append(f"{child_key}={self._scalar_value(item)}")
                    elif self._is_scalar_list(item):
                        grouped_values.append(
                            f"{child_key}={self._join_scalar_list(item)}"
                        )
                    else:
                        nested_lines.extend(
                            self._flatten_grouped_lines(item, child_prefix)
                        )

                output: List[str] = []
                if grouped_values:
                    output.append(f"{prefix}: {' | '.join(grouped_values)}")
                output.extend(nested_lines)
                return output

            root_lines: List[str] = []
            for key, item in value.items():
                child_key = str(key)
                if self._is_scalar(item):
                    root_lines.append(f"{child_key}: {self._scalar_value(item)}")
                elif self._is_scalar_list(item):
                    root_lines.append(f"{child_key}: {self._join_scalar_list(item)}")
                else:
                    root_lines.extend(self._flatten_grouped_lines(item, child_key))
            return root_lines

        if isinstance(value, list):
            if self._is_scalar_list(value):
                joined_values = self._join_scalar_list(value)
                return [f"{prefix}: {joined_values}"] if prefix else [joined_values]

            lines: List[str] = []
            for index, item in enumerate(value):
                child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                if self._is_scalar(item):
                    lines.append(f"{child_prefix}: {self._scalar_value(item)}")
                else:
                    lines.extend(self._flatten_grouped_lines(item, child_prefix))
            return lines

        if prefix:
            return [f"{prefix}: {self._scalar_value(value)}"]
        return [self._scalar_value(value)]

    # ──────────────────────────────────────────────────────────────────────────
    #  Text chunking
    # ──────────────────────────────────────────────────────────────────────────

    def _split_flat_record_text(
        self,
        flat_text: str,
        tokenizer: Any,
        token_limit: int,
    ) -> List[str]:
        normalized = (flat_text or "").strip()
        if not normalized:
            return []

        if self._count_tokens(normalized, tokenizer) <= token_limit:
            return [normalized]

        lines = [line.rstrip() for line in normalized.splitlines() if line.strip()]
        if not lines:
            return []

        prefix_lines = self._detect_identity_lines(lines)
        body_start = len(prefix_lines)
        if not prefix_lines:
            prefix_lines = lines[:1]
            body_start = 1

        body_lines = lines[body_start:]
        parts: List[str] = []
        current_lines = prefix_lines.copy()

        for line in body_lines:
            candidate = "\n".join(current_lines + [line]).strip()

            if (
                len(current_lines) > len(prefix_lines)
                and self._count_tokens(candidate, tokenizer) > token_limit
            ):
                parts.append("\n".join(current_lines).strip())
                current_lines = prefix_lines.copy()
                candidate = "\n".join(current_lines + [line]).strip()

            if self._count_tokens(candidate, tokenizer) > token_limit:
                parts.extend(
                    self._split_oversized_line(
                        prefix_lines=prefix_lines,
                        line=line,
                        tokenizer=tokenizer,
                        token_limit=token_limit,
                    )
                )
 
                current_lines = prefix_lines.copy()
                continue

            current_lines.append(line)

        current_text = "\n".join(current_lines).strip()
        if current_text and (len(current_lines) > len(prefix_lines) or not parts):
            parts.append(current_text)

        return [part for part in parts if part.strip()]

    @staticmethod
    def _detect_identity_lines(lines: List[str]) -> List[str]:
        scored_lines: List[tuple[int, int, str]] = []
        for index, line in enumerate(lines[:8]):
            key = APISourceService._extract_line_key(line)
            score = APISourceService._score_identity_key(key)
            if score > 0:
                scored_lines.append((score, index, line))

        if not scored_lines:
            return lines[: min(2, len(lines))]

        selected_indexes = {
            index
            for _, index, _ in sorted(
                scored_lines,
                key=lambda item: (-item[0], item[1]),
            )[:3]
        }
        return [
            line
            for index, line in enumerate(lines[:8])
            if index in selected_indexes
        ]

    def _split_oversized_line(
        self,
        prefix_lines: List[str],
        line: str,
        tokenizer: Any,
        token_limit: int,
    ) -> List[str]:
        prefix_text = "\n".join(prefix_lines).strip()
        prefix_tokens = self._count_tokens(prefix_text, tokenizer)
        available_tokens = max(32, token_limit - prefix_tokens - 4)
        line_parts = self._split_text_by_token_window(
            text=line,
            tokenizer=tokenizer,
            token_limit=available_tokens,
        )

        output: List[str] = []
        for part in line_parts:
            chunk_lines = prefix_lines + [part] if prefix_lines else [part]
            output.append("\n".join(chunk_lines).strip())
        return output

    def _split_text_by_token_window(
        self,
        text: str,
        tokenizer: Any,
        token_limit: int,
    ) -> List[str]:
        normalized = (text or "").strip()
        if not normalized:
            return []

        token_ids = self._encode_tokens(normalized, tokenizer)
        if token_ids:
            parts: List[str] = []
            position = 0
            while position < len(token_ids):
                window = token_ids[position:position + token_limit]
                piece = self._decode_tokens(window, tokenizer).strip()
                if piece:
                    parts.append(piece)
                position += token_limit
            return parts or [normalized]

        safe_chars = max(128, token_limit * 4)
        return [
            normalized[i:i + safe_chars].strip()
            for i in range(0, len(normalized), safe_chars)
            if normalized[i:i + safe_chars].strip()
        ]

    def _count_tokens(self, text: str, tokenizer: Any) -> int:
        token_ids = self._encode_tokens(text, tokenizer)
        if token_ids:
            return len(token_ids)
        return max(1, len(text) // 4)

    # ──────────────────────────────────────────────────────────────────────────
    #  Scalar / field helpers
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))

    def _is_scalar_list(self, value: Any) -> bool:
        return isinstance(value, list) and all(self._is_scalar(item) for item in value)

    def _join_scalar_list(self, values: List[Any]) -> str:
        return " | ".join(self._scalar_value(value) for value in values)

    @staticmethod
    def _scalar_value(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    @staticmethod
    def _extract_line_key(line: str) -> str:
        if not line:
            return ""
        separator_index = line.find(":")
        if separator_index <= 0:
            return ""
        return line[:separator_index].strip().lower()

    @staticmethod
    def _score_identity_key(key: str) -> int:
        if not key:
            return 0

        normalized = key.lower()
        leaf_key = normalized.rsplit(".", 1)[-1]
        leaf_key = leaf_key.split("[", 1)[0]

        exact_high = {
            "id", "_id", "uuid", "uid", "guid", "key",
            "code", "slug", "identifier", "external_id",
            "reference_id", "ref_id",
        }
        exact_medium = {"name", "title", "label", "number", "ref", "reference"}

        if leaf_key in exact_high:
            return 5
        if leaf_key.endswith("_id") or leaf_key.endswith("id"):
            return 4
        if leaf_key in exact_medium:
            return 3
        if any(
            token in leaf_key
            for token in ("uuid", "guid", "slug", "code", "name", "title", "label")
        ):
            return 2
        return 0

    @staticmethod
    def _score_display_key(key: str) -> int:
        if not key:
            return 0

        normalized = key.lower()
        leaf_key = normalized.rsplit(".", 1)[-1]
        leaf_key = leaf_key.split("[", 1)[0]

        if leaf_key == "display_name":
            return 5
        if leaf_key in {"name", "title", "label"}:
            return 4
        if "name" in leaf_key or "title" in leaf_key or "label" in leaf_key:
            return 2
        return 0

    def _iter_scalar_fields(
        self,
        value: Any,
        prefix: str = "",
    ):
        if isinstance(value, dict):
            for key, item in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from self._iter_scalar_fields(item, child_prefix)
            return

        if isinstance(value, list):
            for index, item in enumerate(value):
                child_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
                yield from self._iter_scalar_fields(item, child_prefix)
            return

        if prefix and self._is_scalar(value) and value not in (None, ""):
            yield prefix, str(value)

    @staticmethod
    def _encode_tokens(text: str, tokenizer: Any) -> List[int]:
        if not tokenizer or not text:
            return []
        try:
            return tokenizer.encode(
                text,
                add_special_tokens=False,
                verbose=False,
            )
        except Exception:
            return []

    @staticmethod
    def _decode_tokens(token_ids: List[int], tokenizer: Any) -> str:
        if not tokenizer or not token_ids:
            return ""
        try:
            return tokenizer.decode(token_ids, skip_special_tokens=True)
        except Exception:
            return ""

    def _resolve_record_identifier(
        self,
        raw_object: Any,
        entry_id: Any,
        record_index: int,
    ) -> Any:
        if entry_id not in (None, ""):
            return entry_id
        if not isinstance(raw_object, dict):
            return f"record-{record_index}"

        best_match: Optional[tuple[int, int, str]] = None
        for key_path, value in self._iter_scalar_fields(raw_object):
            score = self._score_identity_key(key_path)
            if score <= 0:
                continue
            depth = key_path.count(".") + key_path.count("[")
            candidate = str(value)
            if (
                best_match is None
                or score > best_match[0]
                or (score == best_match[0] and depth < best_match[1])
            ):
                best_match = (score, depth, candidate)

        if best_match:
            return best_match[2]

        for _, value in self._iter_scalar_fields(raw_object):
            return value

        return f"record-{record_index}"

    def _resolve_display_name(
        self,
        raw_object: Any,
        entry_id: Any,
        record_index: int,
    ) -> str:
        best_match: Optional[tuple[int, int, str]] = None
        if isinstance(raw_object, dict):
            for key_path, value in self._iter_scalar_fields(raw_object):
                score = self._score_display_key(key_path)
                if score <= 0:
                    continue
                depth = key_path.count(".") + key_path.count("[")
                if (
                    best_match is None
                    or score > best_match[0]
                    or (score == best_match[0] and depth < best_match[1])
                ):
                    best_match = (score, depth, value)
        if best_match:
            return best_match[2]
        if entry_id not in (None, ""):
            return str(entry_id)
        return f"Record {record_index}"
