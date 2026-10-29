"""
json_extractor.py
────────────────────────────────────────────────────────────────────────────────
Responsibility: EXTRACTION ONLY
  • Validate the file (exists, extension, parseable)
  • Load and detect structure (list / envelope dict / single object / JSONL)
  • Serialize each entry to plain text
  • Return an ExtractionResult — a clean list of ExtractedEntry objects

What this file does NOT do:
  • No chunking
  • No token counting
  • No parent/child chunk trees

The ExtractionResult is consumed by JSONChunkingService in json_chunking_service.py

Serialization formats (config["fmt"]):
  "json"      →  pretty-printed JSON   (default)
  "readable"  →  field: value lines, nested indented
  "flat"      →  dot.notation.key: value  (all fields on one level)

Optional extraction config:
  "json_path"   → select a nested portion of the payload before entry parsing
                  Example: "$.data.items"
  "field_paths" → keep only specific fields from each selected entry
                  Example: ["id", "title", "author.name"]

Usage:
    from json_extractor import JSONExtractor

    extractor = JSONExtractor(config={"fmt": "json"})

    # From a file
    result = extractor.extract("nodes.json")
    if result.success:
        for entry in result.entries:
            print(entry.entry_id, len(entry.raw_text))

    # From an already-loaded Python object
    result = extractor.extract_data(my_list, source_name="Nodes")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  SERIALIZER  —  JSON value  →  plain text
# ══════════════════════════════════════════════════════════════════════════════

class JSONSerializer:
    """
    Converts any JSON value to a plain-text string.

    Formats:
      "json"      →  pretty-printed JSON  (default)
      "readable"  →  field: value lines, nested fields indented
      "flat"      →  dot.notation.key: value  (everything on one level)
    """

    FORMATS = ("json", "readable", "flat")

    def __init__(self, fmt: str = "json") -> None:
        if fmt not in self.FORMATS:
            raise ValueError(f"fmt must be one of {self.FORMATS}. Got: {fmt!r}")
        self.fmt = fmt

    def serialize(self, value: Any) -> str:
        if value is None:
            return ""
        if self.fmt == "json":
            return json.dumps(value, indent=2, ensure_ascii=False)
        if self.fmt == "flat":
            return "\n".join(f"{k}: {v}" for k, v in self._flatten(value))
        return self._readable(value, indent=0).strip()

    # ── readable ──────────────────────────────────────────────────────────────

    def _readable(self, node: Any, indent: int) -> str:
        pad, lines = "  " * indent, []
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    lines.append(self._readable(v, indent + 1))
                else:
                    lines.append(f"{pad}{k}: {self._scalar(v)}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}[{i}]:")
                    lines.append(self._readable(item, indent + 1))
                else:
                    lines.append(f"{pad}- {self._scalar(item)}")
        else:
            lines.append(f"{pad}{self._scalar(node)}")
        return "\n".join(lines)

    # ── flat ──────────────────────────────────────────────────────────────────

    def _flatten(self, node: Any, prefix: str = "") -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if isinstance(node, dict):
            for k, v in node.items():
                key = f"{prefix}.{k}" if prefix else k
                pairs.extend(self._flatten(v, key) if isinstance(v, (dict, list)) else [(key, self._scalar(v))])
        elif isinstance(node, list):
            for i, item in enumerate(node):
                key = f"{prefix}[{i}]"
                pairs.extend(self._flatten(item, key) if isinstance(item, (dict, list)) else [(key, self._scalar(item))])
        else:
            pairs.append((prefix or "value", self._scalar(node)))
        return pairs

    @staticmethod
    def _scalar(v: Any) -> str:
        if v is None:           return "null"
        if isinstance(v, bool): return "true" if v else "false"
        return str(v)


# ══════════════════════════════════════════════════════════════════════════════
#  RESULT DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtractedEntry:
    """
    One parsed JSON entry — raw object + serialized text + metadata.
    This is the unit handed off to JSONChunkingService.
    """
    entry_index:    int
    entry_id:       Any          # value of id_field, or None
    display_name:   str          # human label derived from the entry
    raw_object:     Any          # original parsed Python dict/list/scalar
    raw_text:       str          # serialized plain text (ready for chunking)
    serialization_fmt: str
    source_name:    str
    extracted_at:   str


@dataclass
class ExtractionResult:
    """
    Output of JSONExtractor.extract() / .extract_data().
    Passed directly into JSONChunkingService.chunk().
    """
    source_name:    str
    success:        bool
    entries:        List[ExtractedEntry] = field(default_factory=list)
    structure_type: str  = ""
    error:          Optional[str] = None
    metadata:       Dict[str, Any] = field(default_factory=dict)

    # ── convenience ───────────────────────────────────────────────────────────

    def get_entry(self, index: int) -> Optional[ExtractedEntry]:
        return self.entries[index] if 0 <= index < len(self.entries) else None

    def get_entry_by_id(self, entry_id: Any) -> Optional[ExtractedEntry]:
        return next((e for e in self.entries if e.entry_id == entry_id), None)

    def summary(self) -> str:
        lines = [
            f"\n{'─'*60}",
            f"  Source         : {self.source_name}",
            f"  Success        : {self.success}",
            f"  Structure      : {self.structure_type}",
            f"  Entries parsed : {len(self.entries)}",
        ]
        if self.error:
            lines.append(f"  Error          : {self.error}")
        lines.append(f"{'─'*60}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  BASE EXTRACTOR STUB
# ══════════════════════════════════════════════════════════════════════════════

class BaseExtractor:
    """Minimal interface — replace with your actual base_extractor import."""

    def __init__(self, config: Optional[Dict] = None) -> None:
        self.config = config or {}

    def _get_supported_formats(self) -> List[str]:
        raise NotImplementedError

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        raise NotImplementedError


# ══════════════════════════════════════════════════════════════════════════════
#  JSON EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

class JSONExtractor(BaseExtractor):
    """
    Extracts and serializes JSON entries from a file or Python object.
    Produces an ExtractionResult — no chunking involved.

    Config:
        fmt          "json" | "readable" | "flat"  (default: "json")
        id_field     field used as entry identifier (default: "id")
        encoding     file text encoding             (default: "utf-8")
        json_path    optional root selection path   (default: None)
        field_paths  optional per-entry field list  (default: [])
    """

    def __init__(self, config: Optional[Dict] = None) -> None:
        super().__init__(config)
        self.serializer = JSONSerializer(fmt=self.config.get("fmt", "json"))
        self._stats     = {
            "total_calls":         0,
            "successful":          0,
            "failed":              0,
            "validation_failures": 0,
            "total_entries":       0,
        }
        logger.info(
            f"JSONExtractor ready  fmt={self.serializer.fmt!r}  "
            f"id_field={self.config.get('id_field', 'id')!r}  "
            f"json_path={self.config.get('json_path')!r}  "
            f"field_paths={self._get_field_paths()!r}"
        )

    # ── BaseExtractor interface ───────────────────────────────────────────────

    def _get_supported_formats(self) -> List[str]:
        return ["json", "jsonl", "ndjson"]

    @property
    def supported_formats(self) -> List[str]:
        return self._get_supported_formats()

    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Check: exists · supported extension · non-empty · valid JSON/JSONL.
        Returns (is_valid, message).
        """
        path = Path(file_path)

        if not path.exists():
            return False, f"File does not exist: {file_path}"
        if path.suffix.lower() not in (".json", ".jsonl", ".ndjson"):
            return False, f"Unsupported extension {path.suffix!r} — expected .json / .jsonl / .ndjson"
        if path.stat().st_size == 0:
            return False, "File is empty"

        encoding = self.config.get("encoding", "utf-8")
        try:
            raw = path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            return False, f"Encoding error ({encoding}): {exc}"

        if path.suffix.lower() in (".jsonl", ".ndjson"):
            bad: List[str] = []
            for i, line in enumerate(raw.splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    json.loads(line)
                except json.JSONDecodeError as exc:
                    bad.append(f"line {i}: {exc}")
                if len(bad) >= 3:
                    break
            if bad:
                return False, "Invalid JSONL — " + "; ".join(bad)
        else:
            try:
                json.loads(raw)
            except json.JSONDecodeError as exc:
                return False, f"Invalid JSON: {exc}"

        return True, "Valid"

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(
        self,
        file_path:     str,
        source_name:   Optional[str] = None,
        base_metadata: Optional[Dict] = None,
    ) -> Tuple[str, bool, Dict]:
        """
        Load, validate, and serialize all entries in a JSON / JSONL file.

        Returns the standard extractor tuple used throughout the pipeline:
            (markdown_text, success, metadata_dict)

        This matches the interface every other extractor (PDF, DOCX, …) exposes
        so that upload.py can unpack the result as:
            markdown_content, success, extraction_metadata = extractor.extract(path)

        The full ExtractionResult object (entries, raw_objects, etc.) is stored
        on self._last_extraction_result and can be retrieved via extract_result()
        for callers that need it (e.g. JSONChunkingService).
        """
        self._stats["total_calls"] += 1
        path        = Path(file_path)
        source_name = source_name or path.stem

        logger.info(f"[JSONExtractor] extracting: {file_path}")

        # ── Validate ──────────────────────────────────────────────────────────
        is_valid, msg = self.validate_file(file_path)
        if not is_valid:
            self._stats["failed"]              += 1
            self._stats["validation_failures"] += 1
            logger.error(f"  Validation failed: {msg}")
            self._last_extraction_result = ExtractionResult(
                source_name=source_name, success=False, error=msg,
                metadata={"stage": "validation"},
            )
            return "", False, {"error": msg, "stage": "validation"}

        # ── Load ──────────────────────────────────────────────────────────────
        try:
            data, structure_type = self._load_file(path)
        except Exception as exc:
            self._stats["failed"] += 1
            logger.error(f"  Load failed: {exc}")
            self._last_extraction_result = ExtractionResult(
                source_name=source_name, success=False, error=str(exc),
                metadata={"stage": "load"},
            )
            return "", False, {"error": str(exc), "stage": "load"}

        logger.info(f"  Structure: {structure_type!r}  ({len(data)} raw items)")

        # ── Serialize entries ─────────────────────────────────────────────────
        result = self._build_result(data, structure_type, source_name, base_metadata or {})

        if result.success:
            self._stats["successful"]    += 1
            self._stats["total_entries"] += len(result.entries)
        else:
            self._stats["failed"] += 1

        # Store full result for optional retrieval via extract_result()
        self._last_extraction_result = result

        logger.info(
            f"  Extracted {len(result.entries)} entries  "
            f"fmt={self.serializer.fmt!r}"
        )

        # ── Return (text, success, metadata) tuple — standard extractor contract
        if not result.success:
            return "", False, {
                "error": result.error or "Extraction failed",
                "stage": "serialize",
            }

        # Write back as a valid JSON array so the chunking service can
        # call json.loads() on the stored markdown_path content later.
        #
        # Previously this was:
        #   "\n\n".join(e.raw_text for e in result.entries)
        # which produced multiple JSON objects separated by blank lines —
        # NOT a parseable JSON string.  _chunk_json then called json.loads()
        # on that, got a JSONDecodeError, silently fell back to fixed_size
        # chunking, produced chunks with chunk_type="text" instead of
        # "parent"/"child", and the preview loop (which only shows "parent"
        # chunks) returned an empty list even though 30+ chunks were created.
        #
        # Serialising as a JSON array guarantees round-trip fidelity:
        #   upload.py  →  writes array to markdown_path
        #   _chunk_json →  json.loads(content)  →  list  →  structural chunks
        markdown_text = json.dumps(
            [e.raw_object for e in result.entries],
            indent=2,
            ensure_ascii=False,
        )

        metadata: Dict[str, Any] = {
            "extractor":        "JSONExtractor",
            "source_name":      source_name,
            "structure_type":   structure_type,
            "fmt":              self.serializer.fmt,
            "json_path":        self.config.get("json_path"),
            "field_paths":      self._get_field_paths(),
            "entry_count":      len(result.entries),
            "extracted_at":     result.metadata.get("extracted_at", ""),
            # Forward any extra metadata the caller supplied
            **(base_metadata or {}),
        }

        return markdown_text, True, metadata

    def extract_result(
        self,
        file_path:     str,
        source_name:   Optional[str] = None,
        base_metadata: Optional[Dict] = None,
    ) -> ExtractionResult:
        """
        Return the full ExtractionResult object.

        Callers that need the rich entry objects (raw_object, display_name, …)
        — such as JSONChunkingService — should call this method instead of
        extract().  It internally calls extract() to populate the cache and
        then returns self._last_extraction_result.
        """
        # If already called for this file, reuse the cached result
        if (
            hasattr(self, "_last_extraction_result")
            and self._last_extraction_result is not None
            and self._last_extraction_result.source_name == (source_name or Path(file_path).stem)
        ):
            return self._last_extraction_result

        # Otherwise run extraction and return the cached result
        self.extract(file_path, source_name=source_name, base_metadata=base_metadata)
        return self._last_extraction_result  # type: ignore[return-value]

    def extract_data(
        self,
        data:          Union[str, list, dict, Any],
        source_name:   str = "JSON Source",
        base_metadata: Optional[Dict] = None,
    ) -> Tuple[str, bool, Dict]:
        """
        Serialize entries from an already-loaded Python object (no file I/O).

        Returns the same (text, success, metadata) tuple as extract() for
        consistency.  Use extract_data_result() to get the full ExtractionResult.
        """
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as exc:
                self._last_extraction_result = ExtractionResult(
                    source_name=source_name, success=False, error=str(exc),
                    metadata={"stage": "parse"},
                )
                return "", False, {"error": str(exc), "stage": "parse"}

        try:
            structure = "json_list" if isinstance(data, list) else "json_object"
            entries_list, structure = self._prepare_data_for_extraction(data, structure)
            result = self._build_result(entries_list, structure, source_name, base_metadata or {})
        except Exception as exc:
            self._last_extraction_result = ExtractionResult(
                source_name=source_name, success=False, error=str(exc),
                metadata={"stage": "selection"},
            )
            return "", False, {"error": str(exc), "stage": "selection"}

        self._last_extraction_result = result

        if not result.success:
            return "", False, {"error": result.error or "Extraction failed"}

        markdown_text = json.dumps(
            [e.raw_object for e in result.entries],
            indent=2,
            ensure_ascii=False,
        )
        metadata: Dict[str, Any] = {
            "extractor":      "JSONExtractor",
            "source_name":    source_name,
            "structure_type": structure,
            "fmt":            self.serializer.fmt,
            "json_path":      self.config.get("json_path"),
            "field_paths":    self._get_field_paths(),
            "entry_count":    len(result.entries),
        }
        return markdown_text, True, metadata

    def extract_data_result(
        self,
        data:          Union[str, list, dict, Any],
        source_name:   str = "JSON Source",
        base_metadata: Optional[Dict] = None,
    ) -> ExtractionResult:
        """
        Return the full ExtractionResult for an in-memory object.
        Calls extract_data() internally and returns the cached result.
        """
        self.extract_data(data, source_name=source_name, base_metadata=base_metadata)
        return self._last_extraction_result  # type: ignore[return-value]

    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Lightweight file-level info — no serialization or chunking."""
        path = Path(file_path)
        if not path.exists():
            return {"error": "File not found"}

        meta: Dict[str, Any] = {
            "filename":   path.name,
            "extension":  path.suffix.lower(),
            "size_bytes": path.stat().st_size,
            "size_kb":    round(path.stat().st_size / 1024, 2),
        }
        encoding = self.config.get("encoding", "utf-8")
        try:
            raw  = path.read_text(encoding=encoding)
            data = json.loads(raw)
            entries, structure = self._prepare_data_for_extraction(
                data,
                "json_list" if isinstance(data, list) else "json_object",
            )
            meta["structure"] = structure
            meta["entry_count"] = len(entries)
            if isinstance(data, dict):
                meta["top_level_keys"] = list(data.keys())
            if self.config.get("json_path"):
                meta["json_path"] = self.config.get("json_path")
            field_paths = self._get_field_paths()
            if field_paths:
                meta["field_paths"] = field_paths
        except Exception as exc:
            meta["parse_error"] = str(exc)
        return meta

    def get_stats(self) -> Dict[str, Any]:
        total = self._stats["total_calls"]
        if not total:
            return dict(self._stats)
        return {
            **self._stats,
            "success_rate": round(self._stats["successful"] / total * 100, 1),
            "failure_rate": round(self._stats["failed"]     / total * 100, 1),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_file(self, path: Path) -> Tuple[List[Any], str]:
        encoding = self.config.get("encoding", "utf-8")
        raw      = path.read_text(encoding=encoding)

        if path.suffix.lower() in (".jsonl", ".ndjson"):
            lines = [l.strip() for l in raw.splitlines() if l.strip()]
            records = [json.loads(l) for l in lines]
            return self._prepare_data_for_extraction(records, "jsonl")

        data = json.loads(raw)
        return self._prepare_data_for_extraction(
            data,
            "json_list" if isinstance(data, list) else "json_object",
        )

    def _extract_entries(self, data: Any) -> List[Any]:
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("data", "results", "items", "records",
                        "entries", "content", "payload", "rows", "list"):
                val = data.get(key)
                if isinstance(val, list) and val:
                    return val
            return [data]
        return [data]

    def _prepare_data_for_extraction(
        self,
        data: Any,
        base_structure: str,
    ) -> Tuple[List[Any], str]:
        structure = base_structure
        selected_data = data

        json_path = self.config.get("json_path")
        if json_path:
            selected_data, matched = self._apply_path_expression(data, json_path)
            if not matched:
                raise ValueError(f"No data matched json_path '{json_path}'")
            structure = f"{structure}:json_path"

        entries = self._extract_entries(selected_data)

        field_paths = self._get_field_paths()
        if field_paths:
            projected_entries = []
            for entry in entries:
                projected = self._project_entry_fields(entry, field_paths)
                if not self._is_empty_json_value(projected):
                    projected_entries.append(projected)
            if not projected_entries:
                raise ValueError(f"No data matched field_paths {field_paths!r}")
            entries = projected_entries
            structure = f"{structure}:field_projection"

        return entries, structure

    def _get_field_paths(self) -> List[str]:
        raw_field_paths = self.config.get("field_paths") or []
        if isinstance(raw_field_paths, str):
            raw_field_paths = [part.strip() for part in raw_field_paths.split(",")]
        return [str(path).strip() for path in raw_field_paths if str(path).strip()]

    def _apply_path_expression(self, data: Any, path: str) -> Tuple[Any, bool]:
        tokens = self._parse_json_path(path)
        nodes = [data]

        for token_type, token_value in tokens:
            next_nodes: List[Any] = []

            for node in nodes:
                if token_type == "key":
                    if isinstance(node, dict) and token_value in node:
                        next_nodes.append(node[token_value])
                    elif isinstance(node, list):
                        for item in node:
                            if isinstance(item, dict) and token_value in item:
                                next_nodes.append(item[token_value])
                elif token_type == "index":
                    if isinstance(node, list):
                        index = token_value
                        if -len(node) <= index < len(node):
                            next_nodes.append(node[index])
                elif token_type == "wildcard":
                    if isinstance(node, list):
                        next_nodes.extend(node)
                    elif isinstance(node, dict):
                        next_nodes.extend(node.values())

            nodes = next_nodes
            if not nodes:
                return None, False

        if not tokens:
            return data, True
        if len(nodes) == 1:
            return nodes[0], True
        return nodes, True

    def _parse_json_path(self, path: str) -> List[Tuple[str, Any]]:
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
        i = 0

        while i < len(raw_path):
            char = raw_path[i]

            if char == ".":
                i += 1
                start = i
                while i < len(raw_path) and raw_path[i] not in ".[":
                    i += 1
                key = raw_path[start:i].strip()
                if key:
                    tokens.append(("key", key))
                continue

            if char == "[":
                end = raw_path.find("]", i)
                if end == -1:
                    raise ValueError(f"Invalid JSON path {path!r}: missing ']'")
                content = raw_path[i + 1:end].strip()
                if content == "*":
                    tokens.append(("wildcard", "*"))
                elif (
                    len(content) >= 2 and
                    content[0] in {"'", '"'} and
                    content[-1] == content[0]
                ):
                    tokens.append(("key", content[1:-1]))
                else:
                    try:
                        tokens.append(("index", int(content)))
                    except ValueError:
                        tokens.append(("key", content))
                i = end + 1
                continue

            start = i
            while i < len(raw_path) and raw_path[i] not in ".[":
                i += 1
            key = raw_path[start:i].strip()
            if key:
                tokens.append(("key", key))

        return tokens

    def _project_entry_fields(self, entry: Any, field_paths: List[str]) -> Any:
        if not field_paths:
            return entry
        if not isinstance(entry, (dict, list)):
            return entry

        projected: Dict[str, Any] = {}
        for field_path in field_paths:
            value, matched = self._apply_path_expression(entry, field_path)
            if matched:
                self._store_projected_value(projected, field_path, value)
        return projected

    def _store_projected_value(
        self,
        target: Dict[str, Any],
        field_path: str,
        value: Any,
    ) -> None:
        tokens = self._parse_json_path(field_path)
        if not tokens or any(token_type == "wildcard" for token_type, _ in tokens):
            target[field_path] = value
            return

        try:
            container: Any = target
            for index, (token_type, token_value) in enumerate(tokens):
                is_last = index == len(tokens) - 1
                next_type = None if is_last else tokens[index + 1][0]

                if token_type == "key":
                    if not isinstance(container, dict):
                        target[field_path] = value
                        return
                    if is_last:
                        container[token_value] = value
                        return
                    if token_value not in container or container[token_value] is None:
                        container[token_value] = [] if next_type == "index" else {}
                    container = container[token_value]
                    continue

                if token_type == "index":
                    if not isinstance(container, list):
                        target[field_path] = value
                        return
                    while len(container) <= token_value:
                        container.append(None)
                    if is_last:
                        container[token_value] = value
                        return
                    if container[token_value] is None:
                        container[token_value] = [] if next_type == "index" else {}
                    container = container[token_value]
            target[field_path] = value
        except Exception:
            target[field_path] = value

    @staticmethod
    def _is_empty_json_value(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return not value.strip()
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) == 0
        return False

    def _build_result(
        self,
        entries_list:  List[Any],
        structure_type:str,
        source_name:   str,
        base_metadata: Dict,
    ) -> ExtractionResult:
        if not entries_list:
            logger.warning("  No entries found")
            return ExtractionResult(
                source_name=source_name, success=False,
                error="No entries found", structure_type=structure_type,
            )

        id_field = self.config.get("id_field", "id")
        now      = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        result   = ExtractionResult(
            source_name    = source_name,
            success        = True,
            structure_type = structure_type,
        )

        for idx, entry in enumerate(entries_list):
            entry_id     = entry.get(id_field) if isinstance(entry, dict) else None
            display_name = self._display_name(entry, entry_id, idx)
            raw_text     = self.serializer.serialize(entry)

            if not raw_text:
                logger.warning(f"  Entry {idx}: empty after serialization — skipped")
                continue

            result.entries.append(ExtractedEntry(
                entry_index       = idx,
                entry_id          = entry_id,
                display_name      = display_name,
                raw_object        = entry,
                raw_text          = raw_text,
                serialization_fmt = self.serializer.fmt,
                source_name       = source_name,
                extracted_at      = now,
            ))

        if not result.entries:
            result.success = False
            result.error   = "All entries were empty after serialization"

        result.metadata = {
            "source_name":    source_name,
            "structure_type": structure_type,
            "fmt":            self.serializer.fmt,
            "json_path":      self.config.get("json_path"),
            "field_paths":    self._get_field_paths(),
            "entry_count":    len(result.entries),
            "extracted_at":   now,
            **base_metadata,
        }
        return result

    def _display_name(self, entry: Any, entry_id: Any, idx: int) -> str:
        """Derive a human label for breadcrumbs and logging."""
        if isinstance(entry, dict):
            return (
                (entry.get("properties") or {}).get("displayName")
                or entry.get("name")
                or entry.get("title")
                or entry.get("label")
                or (str(entry_id) if entry_id is not None else f"Entry {idx + 1}")
            )
        return f"Entry {idx + 1}"