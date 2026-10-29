"""
Version Service (Stateless)
============================
Manages document versions for scheduled ingestion runs.

Fully stateless — discovers the current version by querying the vector store
for the maximum version_number in the collection. No JSON files, no persistent
storage, no in-memory state. Works correctly with ephemeral K8s CronJob pods.

Version Lifecycle:
    Each CronJob / ingestion run:
      1. Query vector store → find max(version_number) in the collection
      2. New version = max + 1
      3. Deactivate previous version's chunks (is_current_version = false)
      4. Ingest new chunks with new version_number, is_current_version = true
      5. Prune chunks where version_number <= (new_version - MAX_VERSIONS)

    force_reingest:
      1. Query vector store → find max(version_number) = N
      2. Delete all chunks where version_number = N
      3. Re-ingest with version_number = N (same number, replacement)

Retention Policy:
    Only the latest MAX_VERSIONS (default 3) versions are kept.
    Older versions are deleted from the vector store after successful ingestion.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

MAX_VERSIONS = 3


# ---------------------------------------------------------------------------
# Core Functions (stateless — all state lives in the vector store)
# ---------------------------------------------------------------------------

def get_current_version(vector_store, collection: str) -> int:
    """
    Query the vector store to find the max version_number in the collection.

    Returns 0 if no versioned chunks exist (first ingestion).
    """
    backend_name = getattr(vector_store, "BACKEND_NAME", "unknown")

    try:
        if hasattr(vector_store, "get_max_metadata_value"):
            max_version = vector_store.get_max_metadata_value(
                field="version_number", collection=collection
            )
            if max_version is not None:
                return int(max_version)

        elif hasattr(vector_store, "get_distinct_metadata_values"):
            versions = vector_store.get_distinct_metadata_values(
                field="version_number", collection=collection
            )
            if versions:
                return max(int(v) for v in versions if v is not None)

        elif hasattr(vector_store, "scroll_all_metadata"):
            # Fallback: scroll through chunks and find max version
            metadata_list = vector_store.scroll_all_metadata(
                collection=collection, fields=["version_number"], limit=1000
            )
            versions = [
                int(m.get("version_number", 0))
                for m in metadata_list
                if m.get("version_number") is not None
            ]
            if versions:
                return max(versions)

        elif hasattr(vector_store, "count_by_filter"):
            # Binary search approach: check if version N exists
            # Start from a reasonable upper bound and work down
            for v in range(100, 0, -1):
                count = vector_store.count_by_filter(
                    filter_dict={"version_number": v}
                )
                if count and count > 0:
                    return v

        logger.info(
            "[VERSION] Could not determine max version from %s — assuming 0 (first ingestion)",
            backend_name,
        )
        return 0

    except Exception as exc:
        logger.warning(
            "[VERSION] Failed to query max version from %s: %s — assuming 0",
            backend_name, exc,
        )
        return 0


def get_next_version_number(vector_store, collection: str) -> int:
    """Get the next version number (current + 1)."""
    current = get_current_version(vector_store, collection)
    return current + 1


def get_version_for_force_reingest(vector_store, collection: str) -> int:
    """
    Get the version number to use for force_reingest (reuse current version).

    Returns 1 if no versions exist.
    """
    current = get_current_version(vector_store, collection)
    return current if current > 0 else 1


def deactivate_previous_version_chunks(
    vector_store,
    previous_version_number: int,
    collection: str,
) -> int:
    """
    Mark all chunks from a previous version as is_current_version=false.

    Ensures only the latest version's chunks appear in default retrieval.
    Returns the number of updated chunks, or 0 if not supported.
    """
    backend_name = getattr(vector_store, "BACKEND_NAME", "unknown")
    logger.info(
        "[VERSION] Deactivating chunks for version %d | collection=%s | backend=%s",
        previous_version_number, collection, backend_name,
    )

    try:
        if hasattr(vector_store, "update_metadata_by_filter"):
            updated = vector_store.update_metadata_by_filter(
                filter_dict={"version_number": previous_version_number},
                update_fields={"is_current_version": False},
            )
            logger.info(
                "[VERSION] Deactivated %d chunk(s) for version %d",
                updated, previous_version_number,
            )
            return updated

        elif hasattr(vector_store, "set_payload_by_filter"):
            # Qdrant-style: set_payload with filter
            updated = vector_store.set_payload_by_filter(
                filter_dict={"version_number": previous_version_number},
                payload={"is_current_version": False},
            )
            logger.info(
                "[VERSION] Deactivated %d chunk(s) for version %d (set_payload)",
                updated, previous_version_number,
            )
            return updated

        else:
            logger.warning(
                "[VERSION] Backend '%s' does not support bulk metadata update. "
                "Old version chunks (version %d) will retain "
                "is_current_version=true until pruned.",
                backend_name, previous_version_number,
            )
            return 0

    except Exception as exc:
        logger.error(
            "[VERSION] Failed to deactivate version %d chunks: %s",
            previous_version_number, exc,
        )
        return 0


def delete_version_chunks(
    vector_store,
    version_number: int,
    collection: str,
) -> int:
    """
    Delete all chunks belonging to a specific version from the vector store.

    Returns the number of deleted chunks.
    """
    backend_name = getattr(vector_store, "BACKEND_NAME", "unknown")
    logger.info(
        "[VERSION] Deleting chunks for version %d | collection=%s | backend=%s",
        version_number, collection, backend_name,
    )

    try:
        if hasattr(vector_store, "delete_by_filter"):
            deleted = vector_store.delete_by_filter(
                filter_dict={"version_number": version_number}
            )
        elif hasattr(vector_store, "delete_by_metadata"):
            deleted = vector_store.delete_by_metadata(
                key="version_number", value=version_number
            )
        else:
            logger.warning(
                "[VERSION] Backend '%s' has no filter-based deletion. "
                "Chunks for version %d will NOT be deleted.",
                backend_name, version_number,
            )
            return 0

        logger.info(
            "[VERSION] Deleted %d chunk(s) for version %d from '%s'",
            deleted, version_number, collection,
        )
        return deleted

    except Exception as exc:
        logger.error(
            "[VERSION] Failed to delete version %d chunks from '%s': %s",
            version_number, collection, exc,
        )
        return 0


def prune_old_versions(
    vector_store,
    current_version: int,
    collection: str,
    max_versions: int = MAX_VERSIONS,
) -> int:
    """
    Delete chunks from versions older than (current_version - max_versions).

    Example: current_version=4, max_versions=3 → deletes version 1.

    Returns total number of deleted chunks.
    """
    cutoff = current_version - max_versions
    if cutoff <= 0:
        return 0

    total_deleted = 0
    for v in range(1, cutoff + 1):
        deleted = delete_version_chunks(vector_store, v, collection)
        total_deleted += deleted

    if total_deleted > 0:
        logger.info(
            "[VERSION] Pruned versions 1-%d | total deleted=%d | collection=%s",
            cutoff, total_deleted, collection,
        )

    return total_deleted


def get_available_versions(vector_store, collection: str) -> List[Dict[str, Any]]:
    """
    Get all available versions in the collection (for API responses).

    Returns a list of version info dicts discovered from the vector store.
    """
    backend_name = getattr(vector_store, "BACKEND_NAME", "unknown")

    try:
        if hasattr(vector_store, "get_distinct_metadata_values"):
            versions = vector_store.get_distinct_metadata_values(
                field="version_number", collection=collection
            )
            if versions:
                current = max(int(v) for v in versions if v is not None)
                return [
                    {
                        "version_number": int(v),
                        "is_active": int(v) == current,
                    }
                    for v in sorted(versions)
                    if v is not None
                ]
    except Exception as exc:
        logger.warning(
            "[VERSION] Could not get available versions from %s: %s",
            backend_name, exc,
        )

    # Fallback: use current version to infer available versions
    try:
        current = get_current_version(vector_store, collection)
        if current > 0:
            start = max(1, current - MAX_VERSIONS + 1)
            return [
                {
                    "version_number": v,
                    "is_active": v == current,
                }
                for v in range(start, current + 1)
            ]
    except Exception:
        pass

    return []
