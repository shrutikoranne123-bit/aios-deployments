"""
tool_store.py — Persistent store for reusable tool-calling configurations.
===========================================================================

Backs the "saved tool-calling steps" feature. A tool config describes an
external HTTP API (method, URL, auth, params, request body, pagination) that
the smart-retrieval router can call. Unlike the per-request `apis` list on
`/retrieve/smart` (which is throwaway), configs saved here persist to a JSON
file and survive service restarts.

Scoping
-------
Every config is scoped by `org_id` (tenant isolation). Listing / reading /
updating / deleting always takes an `org_id` so one tenant never sees or
touches another tenant's tools.

Storage
-------
A single JSON file (default: `<this_folder>/tool_configs.json`) with shape:

    { "<org_id>": [ {<ToolConfig>}, ... ], ... }

Writes are serialized with a process-level lock and flushed atomically (write
to a temp file, then replace) so a crash mid-write can't corrupt the store.

Auth / pagination models are reused from `services.api_upload` so saved tools
speak the same language as the existing API-source ingestion path.

This module is free of FastAPI imports so it stays unit-testable as a plain
library. The HTTP CRUD endpoints live in retrieval_new.py.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Reuse the mature auth + pagination models already defined for API ingestion
# so saved tools and ingested API sources share one vocabulary.
try:
    # Standalone retrieval deployment: use the vendored copy.
    from retrieval._vendor.api_models import (
        AuthConfig,
        AuthType,
        HttpMethod,
        NoAuth,
        PaginationConfig,
        PaginationType,
    )
except ImportError:
    try:
        from _vendor.api_models import (  # type: ignore
            AuthConfig,
            AuthType,
            HttpMethod,
            NoAuth,
            PaginationConfig,
            PaginationType,
        )
    except ImportError:
        # Monorepo fallback: the original services package.
        from services.api_upload import (  # type: ignore
            AuthConfig,
            AuthType,
            HttpMethod,
            NoAuth,
            PaginationConfig,
            PaginationType,
        )

logger = logging.getLogger(__name__)

# Default storage location: alongside this module so it travels with the
# retrieval service. Overridable via env for container/EFS deployments.
_DEFAULT_STORE_PATH = Path(__file__).resolve().parent / "tool_configs.json"
STORE_PATH = Path(os.getenv("TOOL_STORE_PATH", str(_DEFAULT_STORE_PATH)))


def _json_safe(value: Any) -> Any:
    """Recursively convert a model_dump() result into JSON-serializable data.

    Critically, SecretStr values are unmasked to their real string so the
    stored config can rebuild working auth headers after a restart. Enums are
    reduced to their .value.
    """
    # SecretStr (and any object exposing get_secret_value) -> real string.
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        return getter()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# ===========================================================================
# Model
# ===========================================================================

class ToolConfig(BaseModel):
    """A saved, reusable external-API tool-calling configuration.

    `name` + `description` drive router selection (the LLM router reads the
    description to decide when to call this tool). `url`/`method`/`headers`/
    `query_params`/`json_body` support `{placeholder}` templating filled from
    the router's extracted parameters (always includes `{query}`).
    """

    id: str = Field(
        default_factory=lambda: uuid.uuid4().hex,
        description="Server-assigned unique id. Ignored on create.",
    )
    org_id: str = Field(..., min_length=1, description="Owning tenant id.")
    name: str = Field(
        ..., min_length=1, max_length=100,
        description="Unique tool name the router selects (e.g. 'weather').",
    )
    description: str = Field(
        ..., min_length=1,
        description="What this API returns. The router uses this to pick it.",
    )

    # Request config
    method: HttpMethod = Field(default=HttpMethod.GET, description="HTTP method.")
    url: str = Field(..., min_length=1, description="Full API URL. May contain {placeholders}.")
    headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra request headers (supports {placeholders}).",
    )
    query_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query-string params (supports {placeholders}).",
    )
    json_body: Optional[Dict[str, Any]] = Field(
        default=None,
        description="JSON body for POST/PUT/PATCH (supports {placeholders}).",
    )
    timeout: int = Field(default=20, ge=1, le=120, description="Per-call timeout (seconds).")
    response_path: Optional[str] = Field(
        default=None,
        description="Dotted path into the JSON response to extract, e.g. 'current.temp_c'.",
    )

    # Authentication (discriminated union: none | basic | bearer | api_key | oauth2)
    auth: AuthConfig = Field(
        default_factory=NoAuth,
        description="Authentication strategy and credentials.",
    )

    # Pagination
    pagination: PaginationConfig = Field(
        default_factory=PaginationConfig,
        description="Pagination configuration for multi-page API responses.",
    )

    # Audit
    created_at: Optional[str] = Field(default=None, description="UTC ISO-8601 create time.")
    updated_at: Optional[str] = Field(default=None, description="UTC ISO-8601 last-update time.")

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v

    @field_validator("url", mode="before")
    @classmethod
    def _normalize_url(cls, v: Any) -> Any:
        """Prepend https:// if the scheme is missing (mirrors APISourceInput)."""
        if isinstance(v, str):
            s = v.strip()
            # Keep templated URLs (e.g. "{base}/x") untouched if they start with
            # a placeholder; only auto-scheme plain hostnames.
            if s and not s.startswith(("http://", "https://", "{")):
                s = "https://" + s
            return s
        return v


# ===========================================================================
# Persistence (JSON file, org-scoped, atomic writes)
# ===========================================================================

class ToolStore:
    """JSON-file store of ToolConfig records, scoped by org_id.

    Concurrency
    -----------
    A ``threading.RLock`` serializes access within a single process. For
    multi-worker deployments (e.g. ``uvicorn --workers N``) a best-effort
    cross-process **file lock** additionally serializes the read-modify-write
    of each mutating call, so concurrent workers don't lose each other's
    writes. The atomic temp-file + ``os.replace`` prevents partial-write
    corruption in all cases. If the OS file-lock primitive is unavailable the
    store degrades to in-process locking only (documented limitation).
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self._path = Path(path) if path else STORE_PATH
        self._lock = threading.RLock()
        self._lock_path = self._path.with_suffix(self._path.suffix + ".lock")

    # ---- cross-process file lock ------------------------------------------

    @contextmanager
    def _file_lock(self):
        """Best-effort cross-process exclusive lock around a mutation.

        Uses msvcrt on Windows and fcntl on POSIX. Falls back to a no-op (the
        in-process RLock still applies) if neither is available or the lock
        cannot be acquired, so the store never hard-fails on locking.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lock_file = None
        locked = False
        try:
            lock_file = open(self._lock_path, "a+")
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                    locked = True
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    locked = True
            except Exception as exc:  # locking unavailable — rely on RLock only
                logger.debug("Tool store file lock unavailable: %s", exc)
            yield
        finally:
            if lock_file is not None:
                try:
                    if locked:
                        if os.name == "nt":
                            import msvcrt

                            lock_file.seek(0)
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            import fcntl

                            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                try:
                    lock_file.close()
                except Exception:
                    pass

    # ---- low-level file IO -------------------------------------------------

    def _read_all(self) -> Dict[str, List[Dict[str, Any]]]:
        if not self._path.exists():
            return {}
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                logger.warning("Tool store %s is not an object; ignoring.", self._path)
                return {}
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Failed to read tool store %s: %s", self._path, exc)
            return {}

    def _write_all(self, data: Dict[str, List[Dict[str, Any]]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
        # Atomic replace so a partial write never corrupts the live file.
        os.replace(tmp, self._path)

    @staticmethod
    def _dump(config: ToolConfig) -> Dict[str, Any]:
        """Serialize a ToolConfig to a JSON-safe dict, unmasking SecretStr.

        Pydantic's model_dump_json() renders SecretStr as '**********', which
        would persist masked (unusable) credentials and break tools after a
        restart. We dump in Python mode and walk the structure, replacing any
        SecretStr with its real value so the stored token/password/api_key can
        rebuild a working Authorization header on reload.

        Credentials are therefore stored in PLAINTEXT in the JSON file. This
        matches how vector-store keys already travel in request payloads here;
        secure the file with filesystem permissions in production.
        """
        data = config.model_dump(mode="python")
        return _json_safe(data)

    # ---- CRUD --------------------------------------------------------------

    @staticmethod
    def _name_exists(
        bucket: List[Dict[str, Any]],
        name: str,
        *,
        exclude_id: Optional[str] = None,
    ) -> bool:
        """True if `name` (case/whitespace-insensitive) is taken in this bucket.

        `exclude_id` lets update() ignore the row it is editing so a tool can
        keep its own name. Centralizes the uniqueness rule used by create+update.
        """
        key = (name or "").strip().lower()
        for row in bucket:
            if exclude_id is not None and row.get("id") == exclude_id:
                continue
            if str(row.get("name", "")).strip().lower() == key:
                return True
        return False

    def list(self, org_id: str) -> List[ToolConfig]:
        """Return all configs for a tenant (empty list if none).

        Corrupt/unparseable rows are skipped so one bad row never breaks the
        whole list. When rows are skipped the count is logged at WARNING so a
        schema mismatch (which would otherwise look like "all my tools vanished")
        is visible rather than silent.
        """
        with self._lock:
            raw = self._read_all().get(str(org_id), [])
        out: List[ToolConfig] = []
        skipped = 0
        for item in raw:
            try:
                out.append(ToolConfig(**item))
            except Exception as exc:  # skip corrupt rows, don't fail the list
                skipped += 1
                logger.warning("Skipping unreadable tool config: %s", exc)
        if skipped:
            logger.warning(
                "Tool store: skipped %d of %d config(s) for org=%s that failed to "
                "parse (possible schema mismatch).", skipped, len(raw), org_id,
            )
        return out

    def get(self, org_id: str, tool_id: str) -> Optional[ToolConfig]:
        for cfg in self.list(org_id):
            if cfg.id == tool_id:
                return cfg
        return None

    def get_by_name(self, org_id: str, name: str) -> Optional[ToolConfig]:
        key = (name or "").strip().lower()
        for cfg in self.list(org_id):
            if cfg.name.strip().lower() == key:
                return cfg
        return None

    def create(self, config: ToolConfig) -> ToolConfig:
        """Persist a new config. Raises ValueError if the name already exists."""
        now = datetime.now(timezone.utc).isoformat()
        config.id = uuid.uuid4().hex  # server-assigned; ignore any client id
        config.created_at = now
        config.updated_at = now
        with self._lock, self._file_lock():
            data = self._read_all()
            bucket = data.setdefault(str(config.org_id), [])
            if self._name_exists(bucket, config.name):
                raise ValueError(
                    f"A tool named '{config.name}' already exists for this org."
                )
            bucket.append(self._dump(config))
            self._write_all(data)
        logger.info("Tool config created | org=%s | name=%s | id=%s",
                    config.org_id, config.name, config.id)
        return config

    def update(self, org_id: str, tool_id: str, config: ToolConfig) -> Optional[ToolConfig]:
        """Replace an existing config in place. Returns None if not found."""
        with self._lock, self._file_lock():
            data = self._read_all()
            bucket = data.get(str(org_id), [])
            for idx, row in enumerate(bucket):
                if row.get("id") == tool_id:
                    # Guard against renaming onto another existing tool's name.
                    if self._name_exists(bucket, config.name, exclude_id=tool_id):
                        raise ValueError(
                            f"A tool named '{config.name}' already exists for this org."
                        )
                    config.id = tool_id
                    config.org_id = str(org_id)
                    config.created_at = row.get("created_at")
                    config.updated_at = datetime.now(timezone.utc).isoformat()
                    bucket[idx] = self._dump(config)
                    self._write_all(data)
                    logger.info("Tool config updated | org=%s | id=%s", org_id, tool_id)
                    return config
        return None

    def delete(self, org_id: str, tool_id: str) -> bool:
        """Remove a config. Returns True if something was removed."""
        with self._lock, self._file_lock():
            data = self._read_all()
            bucket = data.get(str(org_id), [])
            new_bucket = [row for row in bucket if row.get("id") != tool_id]
            if len(new_bucket) == len(bucket):
                return False
            if new_bucket:
                data[str(org_id)] = new_bucket
            else:
                data.pop(str(org_id), None)  # drop empty tenant buckets
            self._write_all(data)
        logger.info("Tool config deleted | org=%s | id=%s", org_id, tool_id)
        return True


# ===========================================================================
# ToolConfig  ->  ApiToolSpec dict  (bridge into the existing router/registry)
# ===========================================================================

def to_api_spec_dict(config: ToolConfig) -> Dict[str, Any]:
    """Convert a saved ToolConfig into an `ApiToolSpec.from_dict` payload.

    Applies auth to the outgoing headers/query params and pagination to the
    query params, so the result plugs straight into
    `query_router.make_http_tool` with no special-casing there.
    """
    headers: Dict[str, str] = dict(config.headers or {})
    query_params: Dict[str, Any] = dict(config.query_params or {})

    _apply_auth(config.auth, headers, query_params)
    _apply_pagination(config.pagination, query_params)

    method = config.method.value if hasattr(config.method, "value") else str(config.method)

    return {
        "name": config.name,
        "description": config.description,
        "url": config.url,
        "method": method,
        "headers": headers,
        "query_params": query_params,
        "json_body": config.json_body,
        "timeout": config.timeout,
        "response_path": config.response_path,
    }


def _secret(value: Any) -> str:
    """Read a SecretStr (or plain value) back to its string form."""
    getter = getattr(value, "get_secret_value", None)
    if callable(getter):
        return getter()
    return "" if value is None else str(value)


def _apply_auth(
    auth: Any,
    headers: Dict[str, str],
    query_params: Dict[str, Any],
) -> None:
    """Translate an AuthConfig into request headers (in place).

    - basic  -> Authorization: Basic <base64(user:pass)>
    - bearer -> Authorization: Bearer <token>
    - api key-> <header_name>: <api_key>
    - oauth2 -> not pre-fetched here (left to the caller / future work)
    - none   -> no-op
    """
    if auth is None:
        return
    auth_type = getattr(auth, "type", None)

    if auth_type == AuthType.BEARER:
        headers.setdefault("Authorization", f"Bearer {_secret(auth.token)}")
    elif auth_type == AuthType.BASIC:
        import base64

        raw = f"{auth.username}:{_secret(auth.password)}".encode("utf-8")
        headers.setdefault("Authorization", "Basic " + base64.b64encode(raw).decode("ascii"))
    elif auth_type == AuthType.API_KEY:
        header_name = getattr(auth, "header_name", None) or "X-API-Key"
        headers.setdefault(header_name, _secret(auth.api_key))
    elif auth_type == AuthType.OAUTH2:
        # OAuth2 needs a token-endpoint round trip; not performed at spec-build
        # time. Documented limitation — bearer/basic/api_key are fully wired.
        logger.info(
            "Tool '%s' uses OAuth2; token exchange is not performed when building "
            "the tool spec. Use bearer/basic/api_key for pre-authenticated calls.",
            getattr(auth, "type", "oauth2"),
        )


def _apply_pagination(pagination: Any, query_params: Dict[str, Any]) -> None:
    """Seed the first-page pagination params into the query string (in place).

    Only applies when pagination_type != none. We set the starting page and
    page size so the very first call is correctly paginated; multi-page
    walking is a future enhancement (single-call tools are the common case
    for live-data lookups).
    """
    if pagination is None:
        return
    if getattr(pagination, "pagination_type", PaginationType.NONE) == PaginationType.NONE:
        return

    page_param = getattr(pagination, "page_param_name", None)
    size_param = getattr(pagination, "page_size_param_name", None)
    page_size = getattr(pagination, "page_size_value", None)
    start_page = getattr(pagination, "start_page", 1)

    if page_param and start_page is not None:
        query_params.setdefault(page_param, start_page)
    if size_param and page_size is not None:
        query_params.setdefault(size_param, page_size)


# ===========================================================================
# Module-level singleton
# ===========================================================================

_store: Optional[ToolStore] = None


def get_tool_store() -> ToolStore:
    """Return the process-wide ToolStore singleton."""
    global _store
    if _store is None:
        _store = ToolStore()
    return _store
