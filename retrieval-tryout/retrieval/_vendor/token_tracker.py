"""
Token usage tracker for a single ingestion run.
================================================

Tracks input (prompt) and output (completion) tokens consumed by all model
calls made during one ingestion, from the moment ingestion starts until it
finishes.

It is a process-global singleton because the token-consuming calls (Bedrock
image description, OCR, embeddings) are spread across several services and
threading a tracker object through every call site would be invasive. Access
it via ``get_token_tracker()``.

Usage:
    from services.token_tracker import get_token_tracker

    tracker = get_token_tracker()
    tracker.reset()                       # at the start of a run
    ...
    tracker.add(input_tokens=123, output_tokens=45, source="image_description")
    ...
    totals = tracker.snapshot()           # {"input_tokens": .., "output_tokens": ..}

Recording is best-effort and never raises into the caller: a failure to parse a
usage block must not break ingestion.
"""

import logging
import threading
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class TokenTracker:
    """Thread-safe accumulator of input/output token counts."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._input_tokens = 0
        self._output_tokens = 0
        # Per-source breakdown, e.g. {"image_description": {"input":.., "output":..}}
        self._by_source: Dict[str, Dict[str, int]] = {}

    def reset(self) -> None:
        """Zero all counters. Call at the start of an ingestion run."""
        with self._lock:
            self._input_tokens = 0
            self._output_tokens = 0
            self._by_source = {}

    def add(self, input_tokens: int = 0, output_tokens: int = 0, source: str = "unknown") -> None:
        """
        Add token counts. Best-effort: bad values are coerced/ignored rather
        than raised, so a tracking hiccup never breaks the pipeline.
        """
        try:
            it = int(input_tokens or 0)
            ot = int(output_tokens or 0)
        except (TypeError, ValueError):
            logger.debug("TokenTracker.add got non-numeric values: in=%r out=%r", input_tokens, output_tokens)
            return

        if it <= 0 and ot <= 0:
            return

        with self._lock:
            self._input_tokens += it
            self._output_tokens += ot
            bucket = self._by_source.setdefault(source, {"input_tokens": 0, "output_tokens": 0})
            bucket["input_tokens"] += it
            bucket["output_tokens"] += ot

    def add_from_bedrock_usage(
        self,
        result: dict,
        source: str = "unknown",
        response: Optional[dict] = None,
    ) -> None:
        """
        Extract token usage from a Bedrock invoke_model call and record it.

        Handles the shapes used in this codebase:
          * Anthropic Claude:  body {"usage": {"input_tokens": N, "output_tokens": M}}
          * Cohere (non-Bedrock shape): body meta.billed_units.input_tokens
          * Anything else (e.g. Cohere embeddings on Bedrock, whose body has
            no usage at all): the x-amzn-bedrock-input-token-count /
            -output-token-count response headers. Pass the raw invoke_model
            ``response`` for this fallback.
        Only one source is used per call, so nothing is double-counted.
        Never raises.
        """
        try:
            if isinstance(result, dict):
                usage = result.get("usage")
                if isinstance(usage, dict) and ("input_tokens" in usage or "output_tokens" in usage):
                    self.add(
                        input_tokens=usage.get("input_tokens", 0),
                        output_tokens=usage.get("output_tokens", 0),
                        source=source,
                    )
                    return

                meta = result.get("meta") or {}
                billed = (meta.get("billed_units") if isinstance(meta, dict) else None) or {}
                if isinstance(billed, dict) and billed.get("input_tokens"):
                    self.add(input_tokens=billed.get("input_tokens", 0), output_tokens=0, source=source)
                    return

            if isinstance(response, dict):
                headers = (response.get("ResponseMetadata") or {}).get("HTTPHeaders") or {}
                self.add(
                    input_tokens=headers.get("x-amzn-bedrock-input-token-count", 0),
                    output_tokens=headers.get("x-amzn-bedrock-output-token-count", 0),
                    source=source,
                )
        except Exception as e:  # noqa: BLE001 - tracking must never break ingestion
            logger.debug("TokenTracker.add_from_bedrock_usage failed: %s", e)

    def snapshot(self) -> Dict[str, int]:
        """Return the current totals as a plain dict, including total_tokens."""
        with self._lock:
            return {
                "input_tokens": self._input_tokens,
                "output_tokens": self._output_tokens,
                "total_tokens": self._input_tokens + self._output_tokens,
            }

    def by_source(self) -> Dict[str, Dict[str, int]]:
        """Return the per-source breakdown (copy)."""
        with self._lock:
            return {k: dict(v) for k, v in self._by_source.items()}


# Process-global singleton.
_tracker = TokenTracker()


def get_token_tracker() -> TokenTracker:
    """Return the process-global TokenTracker singleton."""
    return _tracker
