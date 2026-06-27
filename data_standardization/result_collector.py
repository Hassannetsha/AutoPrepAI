"""
result_collector.py
===================
Typed, append-only store for categorical standardization pipeline results.

Design goals
------------
- Single place to read/write results — no scattered dict mutations.
- Typed entry points — callers can't write to the wrong key.
- Append-only log — validation events are never overwritten.
- Snapshot export — `to_dict()` produces a clean dict for downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ------------------------------------------------------------------
# Value objects
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationEvent:
    """A single accept/reject decision recorded during the pipeline."""
    stage: str
    column: str
    action: str
    original_value: str | None = None
    mapped_value: str | None = None
    confidence: float | None = None
    reason: str | None = None
    index: int | None = None
    accepted: bool | None = None


@dataclass
class ColumnStandardizationResult:
    """
    Result of running the cluster pipeline on one categorical column.
    """
    column: str
    accepted_changes: dict[str, str] = field(default_factory=dict)
    total_unique_checked: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    n_clusters: int = 0
    n_auto_clusters: int = 0
    n_ambiguous_clusters: int = 0
    llm_calls: int = 0
    clusters_summary: list[dict[str, Any]] = field(default_factory=list)
    skipped: bool = False
    skip_reason: str | None = None


# ------------------------------------------------------------------
# Collector
# ------------------------------------------------------------------

class ResultCollector:
    """
    Append-only result store for the categorical standardization pipeline.

    One instance per pipeline run. Not thread-safe.
    """

    def __init__(self) -> None:
        # column -> result (set once, never overwritten)
        self._standardization: dict[str, ColumnStandardizationResult] = {}

        # ordered log of every pipeline decision
        self._validation_log: list[ValidationEvent] = []

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def set_column_result(self, result: ColumnStandardizationResult) -> None:
        """
        Record the result for a column.

        Raises
        ------
        ValueError
            If a result for this column has already been recorded.
        """
        if result.column in self._standardization:
            raise ValueError(
                f"Result for column '{result.column}' already recorded. "
                "Create a new ResultCollector for each pipeline run."
            )
        self._standardization[result.column] = result

    def log(self, event: ValidationEvent) -> None:
        self._validation_log.append(event)

    def log_many(self, events: list[ValidationEvent]) -> None:
        self._validation_log.extend(events)

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------

    def column_result(self, column: str) -> ColumnStandardizationResult | None:
        return self._standardization.get(column)

    def all_column_results(self) -> dict[str, ColumnStandardizationResult]:
        return dict(self._standardization)

    def validation_log(self) -> list[ValidationEvent]:
        return list(self._validation_log)

    def accepted_changes(self, column: str) -> dict[str, str]:
        result = self._standardization.get(column)
        return dict(result.accepted_changes) if result else {}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Export results for downstream consumers."""
        return {
            "standardization": {
                col: self._serialize_column_result(r)
                for col, r in self._standardization.items()
            },
            "validation_log": [
                {k: v for k, v in vars(e).items() if v is not None}
                for e in self._validation_log
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_column_result(r: ColumnStandardizationResult) -> dict[str, Any]:
        if r.skipped:
            return {"skipped": True, "skip_reason": r.skip_reason}
        return {
            "accepted_changes": r.accepted_changes,
            "total_unique_checked": r.total_unique_checked,
            "total_accepted": r.total_accepted,
            "total_rejected": r.total_rejected,
            "n_clusters": r.n_clusters,
            "n_auto_clusters": r.n_auto_clusters,
            "n_ambiguous_clusters": r.n_ambiguous_clusters,
            "llm_calls": r.llm_calls,
            "clusters_summary": r.clusters_summary,
        }