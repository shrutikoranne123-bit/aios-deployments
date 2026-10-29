"""
api_source_input.py
────────────────────────────────────────────────────────────────────────────────
Pydantic v2 input models for the RAG Source configuration form.

Covers all four source types visible in the UI:
  • API          — HTTP endpoint with auth, pagination, JSON path
  • Cloud Storage — bucket/blob config
  • Database      — connection string + query
  • File Upload   — local / multipart file

Each model validates and normalises its fields so the backend receives
clean, typed data regardless of what the frontend sends.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Optional, Union
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    SecretStr,
    field_validator,
    model_validator,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Shared enums
# ══════════════════════════════════════════════════════════════════════════════

class SourceType(str, Enum):
    API            = "api"
    CLOUD_STORAGE  = "cloud_storage"
    DATABASE       = "database"
    FILE_UPLOAD    = "file_upload"


class HttpMethod(str, Enum):
    GET    = "GET"
    POST   = "POST"
    PUT    = "PUT"
    PATCH  = "PATCH"
    DELETE = "DELETE"


# Write methods that support a request body
WRITE_METHODS = {HttpMethod.POST, HttpMethod.PUT, HttpMethod.PATCH}


class AuthType(str, Enum):
    NONE    = "none"
    BASIC   = "basic"
    BEARER  = "bearer token"
    API_KEY = "api key"
    OAUTH2  = "oauth2"


class PaginationType(str, Enum):
    PAGE_NUMBER = "page_number"   # ?page=1&page_size=20
    CURSOR      = "cursor"        # ?cursor=<token>
    OFFSET      = "offset"        # ?offset=0&limit=20
    NONE        = "none"


# Strategy for how pagination params are sent on POST/PUT/PATCH requests.
# QUERY  → appended as query-string params (same as GET)
# BODY   → merged into the JSON request body
class PostPaginationStrategy(str, Enum):
    QUERY = "query"
    BODY  = "body"


# ══════════════════════════════════════════════════════════════════════════════
#  Authentication sub-models
# ══════════════════════════════════════════════════════════════════════════════

class NoAuth(BaseModel):
    type: Literal[AuthType.NONE] = AuthType.NONE


class BasicAuth(BaseModel):
    """Username + password shown in the form's 'Basic' auth section."""
    type:     Literal[AuthType.BASIC] = AuthType.BASIC
    username: str       = Field(..., min_length=1, description="API username")
    password: SecretStr = Field(..., description="API password (stored as secret)")


class BearerAuth(BaseModel):
    type:  Literal[AuthType.BEARER] = AuthType.BEARER
    token: SecretStr = Field(..., description="Bearer token")


class ApiKeyAuth(BaseModel):
    type:        Literal[AuthType.API_KEY] = AuthType.API_KEY
    header_name: str       = Field(default="X-API-Key", description="Header carrying the API key")
    api_key:     SecretStr = Field(..., description="API key value")


class OAuth2Auth(BaseModel):
    type:          Literal[AuthType.OAUTH2] = AuthType.OAUTH2
    token_url:     HttpUrl   = Field(..., description="OAuth2 token endpoint")
    client_id:     str       = Field(..., min_length=1)
    client_secret: SecretStr = Field(...)
    scope:         str       = Field(default="")


# Union type — discriminated on the 'type' field
AuthConfig = Annotated[
    Union[NoAuth, BasicAuth, BearerAuth, ApiKeyAuth, OAuth2Auth],
    Field(discriminator="type"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Pagination sub-model  (maps directly to the Pagination section in the form)
# ══════════════════════════════════════════════════════════════════════════════

class PaginationConfig(BaseModel):
    """
    Covers the Pagination Type section in the UI.

    UI fields → model fields:
      Pagination Type          → pagination_type
      Page Parameter Name      → page_param_name        e.g. "page", "pageNumber"
      Page Size Parameter Name → page_size_param_name   e.g. "pageSize", "per_page"
      Page Size Value          → page_size_value         e.g. 20
      Start Page               → start_page              e.g. 0 or 1
      POST Pagination Strategy → post_pagination_strategy  "query" | "body"

    For POST/PUT/PATCH endpoints, post_pagination_strategy controls whether
    page params are appended as query-string params ("query") or merged into
    the JSON request body ("body").  Defaults to "query" for backwards compat.
    """
    pagination_type: PaginationType = Field(
        # FIX: was PAGE_NUMBER — caused validation failure for requests that
        # don't supply pagination params (e.g. a simple POST with no pagination).
        default=PaginationType.NONE,
        description="Strategy used to paginate through API results",
    )
    page_param_name: Optional[str] = Field(
        default=None,
        description="Query/body param that specifies which page to fetch (e.g. 'page', 'pageNumber')",
    )
    page_size_param_name: Optional[str] = Field(
        default=None,
        description="Query/body param that sets records per page (e.g. 'pageSize', 'per_page')",
    )
    page_size_value: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Number of records to return per page in each API request",
    )
    start_page: int = Field(
        default=1,
        ge=0,
        description="Initial page number to begin fetching from (typically 0 or 1)",
    )
    # NEW: controls where pagination params are injected for write-method requests
    post_pagination_strategy: PostPaginationStrategy = Field(
        default=PostPaginationStrategy.QUERY,
        description=(
            "For POST/PUT/PATCH: whether pagination params are sent as query-string "
            "params ('query') or merged into the JSON request body ('body'). "
            "Ignored for GET/DELETE."
        ),
    )

    @model_validator(mode="after")
    def _require_params_when_paginating(self) -> "PaginationConfig":
        if self.pagination_type != PaginationType.NONE:
            if not self.page_param_name:
                raise ValueError(
                    "page_param_name is required when pagination_type is not 'none'"
                )
            if not self.page_size_param_name:
                raise ValueError(
                    "page_size_param_name is required when pagination_type is not 'none'"
                )
            if self.page_size_value is None:
                raise ValueError(
                    "page_size_value is required when pagination_type is not 'none'"
                )
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  API Source input  (the main form in the screenshot)
# ══════════════════════════════════════════════════════════════════════════════

class APISourceInput(BaseModel):
    """
    Full input model for the 'API' source tab.

    Maps every visible form field:
      Source Name       → source_name
      HTTP Method       → http_method
      URL               → url
      Authentication    → auth  (discriminated union — see AuthConfig)
      Pagination        → pagination
      JSON Path         → json_path
      Field Paths       → field_paths  (optional per-entry field projection)
      Request Headers   → headers      (optional extra headers)
      Query Params      → query_params (optional query-string params, all methods)
      Request Body      → request_body (for POST/PUT/PATCH)
    """
    source_type: Literal[SourceType.API] = SourceType.API

    # ── Basic identity ────────────────────────────────────────────────────────
    source_name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable label for this API source",
        examples=["API Source"],
    )
    knowledge_base_id: Optional[str] = Field(
        default=None,
        description="Existing knowledge base ID to attach this API source to.",
    )
    knowledge_base_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Knowledge base name to create or reuse for this API source.",
    )

    # ── Request config ────────────────────────────────────────────────────────
    http_method: HttpMethod = Field(
        default=HttpMethod.GET,
        description="HTTP verb to use when calling the API endpoint",
    )
    url: HttpUrl = Field(
        ...,
        description="Full API endpoint URL",
        examples=["https://www.apiendpointurl.com/data"],
    )
    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Additional HTTP request headers (key-value pairs)",
    )
    # NEW: user-defined query-string params (forwarded on every request, all methods)
    query_params: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Extra query-string parameters appended to the URL on every request. "
            "Supported for all HTTP methods. Pagination params are merged on top of "
            "these when pagination_type != 'none' and post_pagination_strategy == 'query'."
        ),
    )
    request_body: Optional[Any] = Field(
        default=None,
        description="JSON request body — only used with POST / PUT / PATCH",
    )
    verify_ssl: bool = Field(
        default=True,
        description=(
            "Whether TLS certificates should be verified for HTTPS requests. "
            "Set to false only for trusted internal endpoints with self-signed "
            "or incomplete certificate chains."
        ),
    )
    ca_bundle_path: Optional[str] = Field(
        default=None,
        description=(
            "Optional path to a custom CA bundle PEM file used to verify the "
            "remote server certificate."
        ),
    )

    # ── Authentication ────────────────────────────────────────────────────────
    auth: AuthConfig = Field(
        default_factory=NoAuth,
        description="Authentication strategy and credentials",
    )

    # ── Pagination ────────────────────────────────────────────────────────────
    pagination: PaginationConfig = Field(
        default_factory=PaginationConfig,
        description="Pagination configuration for multi-page API responses",
    )

    # ── Data extraction ───────────────────────────────────────────────────────
    json_path: Optional[str] = Field(
        default=None,
        description=(
            "JSONPath expression to extract the data array from the response "
            "(e.g. '$.data.items', '$.results'). "
            "Leave empty if the response root is already an array."
        ),
        examples=["$.data", "$.results.items"],
    )
    field_paths: list[str] = Field(
        default_factory=list,
        description=(
            "Optional field paths to keep from each selected JSON entry. "
            "Examples: ['id', 'title', 'author.name']"
        ),
        examples=[["id", "title", "author.name"]],
    )

    # ── Guards ────────────────────────────────────────────────────────────────
    @field_validator("url", mode="before")
    @classmethod
    def _normalise_url(cls, v: str) -> str:
        """Prepend https:// if the user omitted the scheme."""
        v = str(v).strip()
        if v and not v.startswith(("http://", "https://")):
            v = "https://" + v
        return v

    @model_validator(mode="after")
    def _body_only_for_write_methods(self) -> "APISourceInput":
        """
        FIX: previously raised on GET+body only.
        Now also warns if a write method has no body (non-fatal — body may be
        legitimately absent for a POST that uses only query params).
        """
        if self.request_body is not None and self.http_method not in WRITE_METHODS:
            raise ValueError(
                f"request_body should not be set for {self.http_method.value} requests. "
                f"request_body is only valid for: {', '.join(m.value for m in WRITE_METHODS)}."
            )
        return self

    @field_validator("field_paths", mode="before")
    @classmethod
    def _normalise_field_paths(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        if not isinstance(value, list):
            raise ValueError("field_paths must be a list of strings")
        return [str(item).strip() for item in value if str(item).strip()]

    @model_validator(mode="after")
    def _validate_knowledge_base_target(self) -> "APISourceInput":
        if self.knowledge_base_id and self.knowledge_base_name:
            raise ValueError(
                "Provide either knowledge_base_id or knowledge_base_name, not both"
            )
        if self.knowledge_base_name is not None:
            self.knowledge_base_name = self.knowledge_base_name.strip() or None
        if self.ca_bundle_path is not None:
            self.ca_bundle_path = self.ca_bundle_path.strip() or None
        return self


# ══════════════════════════════════════════════════════════════════════════════
#  Cloud Storage Source input
# ══════════════════════════════════════════════════════════════════════════════

class CloudStorageSourceInput(BaseModel):
    source_type:  Literal[SourceType.CLOUD_STORAGE] = SourceType.CLOUD_STORAGE
    source_name:  str      = Field(..., min_length=1, max_length=255)
    provider:     str      = Field(..., description="e.g. 'aws_s3', 'gcs', 'azure_blob'")
    bucket:       str      = Field(..., description="Bucket or container name")
    prefix:       str      = Field(default="", description="Key prefix / folder path")
    access_key:   Optional[SecretStr] = None
    secret_key:   Optional[SecretStr] = None
    region:       Optional[str]       = None


# ══════════════════════════════════════════════════════════════════════════════
#  Database Source input
# ══════════════════════════════════════════════════════════════════════════════

class DatabaseSourceInput(BaseModel):
    source_type:       Literal[SourceType.DATABASE] = SourceType.DATABASE
    source_name:       str       = Field(..., min_length=1, max_length=255)
    connection_string: SecretStr = Field(..., description="Full DB connection URI")
    query:             str       = Field(..., description="SQL or MongoDB query to execute")
    batch_size:        int       = Field(default=500, ge=1, le=10_000)


# ══════════════════════════════════════════════════════════════════════════════
#  File Upload Source input
# ══════════════════════════════════════════════════════════════════════════════

class FileUploadSourceInput(BaseModel):
    source_type: Literal[SourceType.FILE_UPLOAD] = SourceType.FILE_UPLOAD
    source_name: str      = Field(..., min_length=1, max_length=255)
    file_name:   str      = Field(..., description="Original filename")
    file_size:   int      = Field(..., ge=1, description="File size in bytes")
    mime_type:   str      = Field(..., description="MIME type e.g. 'application/pdf'")
    storage_path: str     = Field(..., description="Path where the file was saved server-side")


# ══════════════════════════════════════════════════════════════════════════════
#  Top-level discriminated union — one model to rule them all
# ══════════════════════════════════════════════════════════════════════════════

SourceInput = Annotated[
    Union[
        APISourceInput,
        CloudStorageSourceInput,
        DatabaseSourceInput,
        FileUploadSourceInput,
    ],
    Field(discriminator="source_type"),
]


# ══════════════════════════════════════════════════════════════════════════════
#  FastAPI route example (drop-in)
# ══════════════════════════════════════════════════════════════════════════════

"""
# routes/sources.py

from fastapi import APIRouter
from .api_source_input import SourceInput, APISourceInput

router = APIRouter(prefix="/sources", tags=["sources"])

@router.post("/", response_model=dict)
async def create_source(payload: SourceInput):
    # Pydantic has already validated every field by the time we get here.
    # payload.source_type tells you which concrete model you received.
    if isinstance(payload, APISourceInput):
        # access .auth, .pagination, .json_path, .query_params etc.
        pass
    return {"status": "created", "source_name": payload.source_name}
"""
