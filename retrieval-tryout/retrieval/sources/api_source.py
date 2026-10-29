"""
API Source Handler
Supports legacy request_data dict → converts to APISourceInput model
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    from retrieval._vendor.api_models import (
        APISourceInput,
        AuthType,
        HttpMethod,
        NoAuth,
        BasicAuth,
        BearerAuth,
        ApiKeyAuth,
        PaginationConfig,
        PaginationType,
        PostPaginationStrategy,
        SourceType,
    )
except ImportError:
    from services.api_upload import (
        APISourceInput,
        AuthType,
        HttpMethod,
        NoAuth,
        BasicAuth,
        BearerAuth,
        ApiKeyAuth,
        PaginationConfig,
        PaginationType,
        PostPaginationStrategy,
        SourceType,
    )
try:
    from retrieval._vendor.api_source_service import APISourceService
except ImportError:
    from services.api_source_service import APISourceService

logger = logging.getLogger(__name__)


class APISource:
    """Handle API data fetching with proper request data extraction.

    Supports GET and all write methods (POST, PUT, PATCH).

    POST pagination strategy
    ────────────────────────
    Controlled by ``source_details.pagination.post_pagination_strategy``:
      "query" (default) — pagination params go to the query-string (same as GET)
      "body"            — pagination params are merged into the JSON request body

    This is only relevant when pagination_type != "none" AND the HTTP method
    is POST, PUT, or PATCH.
    """

    @staticmethod
    def _sanitize_failed_filename_component(value: Any, default: str) -> str:
        """Convert API identifiers into safe filename components."""
        import re

        raw_value = str(value or "").strip()
        if not raw_value:
            return default

        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", raw_value).strip("._-")
        return normalized or default

    @classmethod
    def build_discovery_failed_filename(cls, source_details: Dict[str, Any]) -> str:
        """Build the failed-files entry for API discovery failures."""
        method_name = cls._sanitize_failed_filename_component(
            source_details.get("method"),
            "get",
        ).lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"api_{method_name}_{timestamp}"

    def _build_output_filename(self) -> str:
        """Build the visible filename for successful API ingestion outputs."""
        return self.build_discovery_failed_filename(self.source_details)

    def __init__(self, request_data: Dict[str, Any]):
        self.request_data = request_data
        self.source_details = request_data.get("source_details", {})
        self.payload: APISourceInput = self._extract_request_info()

        logger.info(
            "API Source initialized: %s %s",
            self.payload.http_method,
            self.payload.url,
        )

    def _extract_request_info(self) -> APISourceInput:
        source_details = self.source_details

        source_name = (
            source_details.get("source_name")
            or source_details.get("name")
            or self.request_data.get("name")
            or "API Source"
        )

        url = source_details.get("url", "")
        method_str = source_details.get("method", "GET").upper()
        try:
            http_method = HttpMethod(method_str)
        except ValueError:
            http_method = HttpMethod.GET
            logger.warning("Invalid HTTP method '%s', defaulting to GET", method_str)

        auth_config = self._extract_auth_config()
        pagination_config = self._extract_pagination_config()
        headers = self._parse_headers()

        # FIX: query_params are now forwarded to APISourceInput instead of
        # being silently discarded.
        query_params = self._parse_query_params()

        request_body, json_path = self._resolve_request_body_and_json_path(
            source_details=source_details,
            http_method=http_method,
        )
        verify_ssl = self._extract_verify_ssl(source_details)
        ca_bundle_path = self._extract_ca_bundle_path(source_details)
        field_paths = source_details.get("field_paths", [])
        if isinstance(field_paths, str):
            field_paths = [p.strip() for p in field_paths.split(",") if p.strip()]

        knowledge_base_id = source_details.get("knowledge_base_id")
        knowledge_base_name = source_details.get("knowledge_base_name")

        try:
            return APISourceInput(
                source_type=SourceType.API,
                source_name=source_name,
                knowledge_base_id=knowledge_base_id,
                knowledge_base_name=knowledge_base_name,
                http_method=http_method,
                url=url,
                headers=headers,
                query_params=query_params,       # FIX: was silently dropped before
                request_body=request_body,
                verify_ssl=verify_ssl,
                ca_bundle_path=ca_bundle_path,
                auth=auth_config,
                pagination=pagination_config,
                json_path=json_path,
                field_paths=field_paths,
            )
        except Exception as exc:
            logger.error("Failed to create APISourceInput: %s", exc, exc_info=True)
            raise ValueError(f"Invalid API source configuration: {exc}") from exc

    def _resolve_request_body_and_json_path(
        self,
        source_details: Dict[str, Any],
        http_method: HttpMethod,
    ) -> Tuple[Any, Optional[str]]:
        """Resolve request_body and json_path with a POST compatibility fallback."""
        json_path = source_details.get("json_path")

        # request_body is only meaningful for write methods; the Pydantic
        # validator on APISourceInput will reject it for GET/DELETE.
        explicit_request_body = (
            source_details.get("request_body")
            if source_details.get("request_body") is not None
            else source_details.get("body")
        )
        if explicit_request_body is not None:
            return explicit_request_body, json_path

        if http_method not in {HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH}:
            return None, json_path

        inferred_request_body = self._parse_jsonish_body(json_path)
        if inferred_request_body is None:
            return None, json_path

        logger.warning(
            "Write-method API source received a JSON object in source_details.json_path "
            "without request_body/body. Treating that value as the request body and "
            "defaulting response json_path to '$'."
        )
        return inferred_request_body, "$"

    def _parse_jsonish_body(self, candidate: Any) -> Optional[Any]:
        """Parse a JSON-ish object/array string used as a misplaced request body."""
        if not isinstance(candidate, str):
            return None

        raw_value = candidate.strip()
        if not raw_value:
            return None

        if raw_value.startswith("`") and raw_value.endswith("`"):
            raw_value = raw_value[1:-1].strip()

        if not raw_value or raw_value[0] not in "[{":
            return None

        try:
            return json.loads(raw_value)
        except json.JSONDecodeError:
            pass

        normalized = self._remove_trailing_commas(
            self._strip_line_comments(raw_value)
        )
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            logger.warning(
                "Could not parse source_details.json_path as a JSON request body fallback."
            )
            return None

    @staticmethod
    def _strip_line_comments(text: str) -> str:
        """Remove // comments while preserving string contents."""
        output: List[str] = []
        in_string = False
        string_delimiter = ""
        escape = False
        index = 0

        while index < len(text):
            char = text[index]
            next_char = text[index + 1] if index + 1 < len(text) else ""

            if escape:
                output.append(char)
                escape = False
                index += 1
                continue

            if in_string:
                output.append(char)
                if char == "\\":
                    escape = True
                elif char == string_delimiter:
                    in_string = False
                index += 1
                continue

            if char in {'"', "'"}:
                in_string = True
                string_delimiter = char
                output.append(char)
                index += 1
                continue

            if char == "/" and next_char == "/":
                index += 2
                while index < len(text) and text[index] not in "\r\n":
                    index += 1
                continue

            output.append(char)
            index += 1

        return "".join(output)

    @staticmethod
    def _remove_trailing_commas(text: str) -> str:
        """Remove trailing commas before closing braces/brackets outside strings."""
        output: List[str] = []
        in_string = False
        string_delimiter = ""
        escape = False
        index = 0

        while index < len(text):
            char = text[index]

            if escape:
                output.append(char)
                escape = False
                index += 1
                continue

            if in_string:
                output.append(char)
                if char == "\\":
                    escape = True
                elif char == string_delimiter:
                    in_string = False
                index += 1
                continue

            if char in {'"', "'"}:
                in_string = True
                string_delimiter = char
                output.append(char)
                index += 1
                continue

            if char == ",":
                look_ahead = index + 1
                while look_ahead < len(text) and text[look_ahead] in " \t\r\n":
                    look_ahead += 1
                if look_ahead < len(text) and text[look_ahead] in "}]":
                    index += 1
                    continue

            output.append(char)
            index += 1

        return "".join(output)

    def _extract_auth_config(self):
        """Support both legacy and new structure with auth_details."""
        source_details = self.source_details

        auth_type_str = source_details.get("auth_type", "none").lower()
        try:
            auth_type = AuthType(auth_type_str)
        except ValueError:
            auth_type = AuthType.NONE

        # New structure: auth_details object
        auth_details = source_details.get("auth_details") or {}

        if auth_type == AuthType.NONE:
            return NoAuth()

        elif auth_type == AuthType.BASIC:
            username = auth_details.get("username") or source_details.get("username", "")
            password = auth_details.get("password") or source_details.get("password", "")
            return BasicAuth(username=username, password=password)

        elif auth_type == AuthType.BEARER:
            token = (
                auth_details.get("token")
                or auth_details.get("bearer_token")
                or source_details.get("token")
                or source_details.get("bearer_token", "")
            )
            return BearerAuth(token=token)

        elif auth_type == AuthType.API_KEY:
            return ApiKeyAuth(
                header_name=(
                    auth_details.get("header_name")
                    or source_details.get("api_key_header", "X-API-Key")
                ),
                api_key=(
                    auth_details.get("api_key")
                    or source_details.get("api_key", "")
                ),
            )

        return NoAuth()

    def _extract_pagination_config(self) -> PaginationConfig:
        """Extract pagination config from both current and legacy request shapes.

        For POST/PUT/PATCH endpoints, the optional ``post_pagination_strategy``
        field controls whether pagination params are sent as query-string params
        ("query", default) or merged into the JSON request body ("body").
        """
        pagination_data = self.source_details.get("pagination", {})
        pagination_type_str = pagination_data.get("type", "none").lower()

        try:
            pagination_type = PaginationType(pagination_type_str)
        except ValueError:
            pagination_type = PaginationType.NONE

        # Parse post_pagination_strategy (only meaningful for write methods)
        post_strategy_str = (
            pagination_data.get("post_pagination_strategy", "query").lower()
        )
        try:
            post_pagination_strategy = PostPaginationStrategy(post_strategy_str)
        except ValueError:
            post_pagination_strategy = PostPaginationStrategy.QUERY
            logger.warning(
                "Invalid post_pagination_strategy '%s', defaulting to 'query'",
                post_strategy_str,
            )

        return PaginationConfig(
            pagination_type=pagination_type,
            page_param_name=(
                pagination_data.get("page_param_name")
                or pagination_data.get("current_page_name")
                or pagination_data.get("page_name")
            ),
            page_size_param_name=(
                pagination_data.get("page_size_param_name")
                or pagination_data.get("page_size_name")
                or pagination_data.get("size_name")
            ),
            page_size_value=(
                pagination_data.get("page_size_value")
                or pagination_data.get("size")
            ),
            start_page=self._safe_int(
                pagination_data.get("start_page")
                or pagination_data.get("current_page_value")
                or pagination_data.get("page"),
                1,
            ),
            post_pagination_strategy=post_pagination_strategy,
        )

    def _parse_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        headers_data = self.source_details.get("headers", [])

        if isinstance(headers_data, list):
            for item in headers_data:
                if isinstance(item, dict):
                    key = item.get("key", "").strip()
                    value = item.get("value", "").strip()
                    if key and value:
                        headers[key] = value
        elif isinstance(headers_data, dict):
            headers = {k: str(v) for k, v in headers_data.items() if k and v}

        return headers

    def _parse_query_params(self) -> Dict[str, str]:
        """Parse user-defined query params from source_details.

        These are forwarded to APISourceInput.query_params and sent on every
        request regardless of HTTP method or pagination strategy.
        """
        params: Dict[str, str] = {}
        params_data = self.source_details.get("query_params", [])

        if isinstance(params_data, list):
            for item in params_data:
                if isinstance(item, dict):
                    key = item.get("key", "").strip()
                    value = item.get("value", "").strip()
                    if key and value:
                        params[key] = value
        elif isinstance(params_data, dict):
            params = {k: str(v) for k, v in params_data.items() if k and v}

        return params

    @staticmethod
    def _extract_verify_ssl(source_details: Dict[str, Any]) -> bool:
        """Resolve TLS verification from legacy request shapes."""
        raw_value = (
            source_details.get("verify_ssl")
            if source_details.get("verify_ssl") is not None
            else source_details.get("verifySsl")
        )
        if raw_value is None:
            raw_value = (
                source_details.get("ssl_verify")
                if source_details.get("ssl_verify") is not None
                else source_details.get("sslVerify")
            )
        if raw_value is None:
            raw_value = source_details.get("verify")

        if raw_value is None:
            return True

        if isinstance(raw_value, bool):
            return raw_value

        return str(raw_value).strip().lower() not in {"0", "false", "no", "off"}

    @staticmethod
    def _extract_ca_bundle_path(source_details: Dict[str, Any]) -> Optional[str]:
        """Resolve a custom CA bundle path from legacy request shapes."""
        for key in (
            "ca_bundle_path",
            "caBundlePath",
            "ca_cert_path",
            "caCertPath",
            "ssl_cert_path",
            "sslCertPath",
        ):
            value = source_details.get(key)
            if value is None:
                continue

            normalized = str(value).strip()
            if normalized:
                return normalized

        return None

    @staticmethod
    def _safe_int(value: Any, default: int) -> int:
        try:
            if value in (None, ""):
                return default
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _with_row_indexes(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        indexed_rows: List[Dict[str, Any]] = []
        for index, row in enumerate(rows, start=1):
            row_payload = dict(row)
            row_payload["row_index"] = index
            indexed_rows.append(row_payload)
        return indexed_rows

    async def discover(self) -> List[Dict[str, Any]]:
        """Discover API content and prepare it for ingestion."""
        try:
            pag = self.payload.pagination
            requested_chunking_type = (
                self.request_data.get("chunking_details", {}).get("chunking_type")
                or self.request_data.get("chunking_type")
                or "DEFAULT"
            )
            chunking_type_upper = str(requested_chunking_type).strip().upper()

            logger.info("=" * 60)
            logger.info("🚀 STARTING PAGINATED API INGESTION")
            logger.info("=" * 60)
            logger.info(f"URL                       : {self.payload.url}")
            logger.info(f"HTTP Method               : {self.payload.http_method.value}")
            logger.info(f"Pagination Type           : {pag.pagination_type}")
            logger.info(f"Page Param                : {pag.page_param_name}")
            logger.info(f"Page Size Param           : {pag.page_size_param_name}")
            logger.info(f"Page Size Value           : {pag.page_size_value}")
            logger.info(f"Start Page                : {pag.start_page}")
            logger.info(f"POST Pagination Strategy  : {pag.post_pagination_strategy}")
            logger.info(f"JSON Path                 : {self.payload.json_path}")
            logger.info(f"Chunking Type             : {chunking_type_upper}")
            logger.info(f"Query Params              : {self.payload.query_params}")
            logger.info(f"Has Request Body          : {self.payload.request_body is not None}")

            service = APISourceService()
            service.DEFAULT_TIMEOUT_SECONDS = 60.0
            service.MAX_PAGES = 30

            result = await service.fetch_and_extract(
                self.payload,
                self.request_data.get("embedding_details", {}),
            )

            entries_count = result.get("entries_extracted", 0)
            pages_fetched = result.get("pages_fetched", 0)

            logger.info("=" * 60)
            logger.info("✅ PAGINATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Pages Fetched    : {pages_fetched}")
            logger.info(f"Total Records    : {entries_count}")
            logger.info(f"Data Extracted   : {'SUCCESS' if entries_count > 0 else 'FAILED'}")

            if entries_count > 0 and result.get("data"):
                sample = result["data"][0]
                logger.info(
                    "Sample Record    : %s | %s...",
                    sample.get("book_id"),
                    str(sample.get("title", ""))[:80],
                )

            output_filename = self._build_output_filename()
            # content_size_bytes is the byte length of the full aggregated
            # JSON response across ALL pages, calculated in fetch_and_extract.
            content_size_bytes = result.get("content_size_bytes", 0)
            logger.info(
                "API content size: %s bytes across %s page(s)",
                content_size_bytes,
                pages_fetched,
            )

            if chunking_type_upper == "JSON":
                return [{
                    "filename": output_filename,
                    "file_path": None,
                    "file_type": ".json",
                    "content": result["json_chunk_text"],
                    "source": "api",
                    "api_url": str(self.payload.url),
                    "record_count": entries_count,
                    "pages_fetched": pages_fetched,
                    "source_name": self.payload.source_name,
                    "content_size_bytes": content_size_bytes,
                }]

            flat_rows = self._with_row_indexes(result.get("flat_rows", []))
            logger.info("Flat Rows Prepared: %s", len(flat_rows))

            if not flat_rows:
                raise ValueError("No flattened API rows were prepared for ingestion")

            logger.info(
                "Using flattened API row-wise ingestion for non-JSON chunking mode; "
                "chunking settings are skipped."
            )

            return [{
                "filename": output_filename,
                "file_path": None,
                "file_type": ".dbrows",
                "rows": flat_rows,
                "db_type": "api",
                "source": "api",
                "api_url": str(self.payload.url),
                "record_count": entries_count,
                "pages_fetched": pages_fetched,
                "source_name": self.payload.source_name,
                "content_size_bytes": content_size_bytes,
            }]

        except Exception as exc:
            if "CERTIFICATE_VERIFY_FAILED" in str(exc):
                logger.error(
                    "TLS verification failed for API source '%s'. "
                    "Provide source_details.ca_bundle_path (preferred) or set "
                    "source_details.verify_ssl=false only for trusted internal endpoints.",
                    self.payload.source_name,
                )
            logger.error("❌ API discovery failed: %s", exc, exc_info=True)
            raise
        
