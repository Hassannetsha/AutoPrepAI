"""
detection_strategy.py
=====================
Abstract interface for categorical inconsistency detection strategies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class ColumnIssue:
    """
    A single detected issue in a column value.

    Attributes
    ----------
    index     : Row index in the dataframe. None = value-level issue
                (e.g. a categorical variant affects all rows with that value).
    value     : The problematic value.
    confidence: 0.0–1.0.
    reason    : Human-readable explanation.
    kind      : Strategy-specific tag e.g. "variant", "llm_flagged".
    canonical : The target value this should map to. None if no remapping suggested.
    """
    index: int | None
    value: Any
    confidence: float
    reason: str
    kind: str = "unknown"
    canonical: str | None = None


@dataclass(frozen=True)
class DetectionContext:
    """
    Shared context passed to every strategy on each call.
    """
    column: str
    allowed_values: Optional[list[str]] = None  # ← CHANGED: None triggers discovery mode
    rules: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DetectionResult:
    """
    Unified return type for all DetectionStrategy implementations.
    """
    issues: list[ColumnIssue]
    log: list[dict] = field(default_factory=list)


class DetectionStrategy(ABC):

    @abstractmethod
    def detect(
        self,
        series: pd.Series,
        context: DetectionContext,
    ) -> DetectionResult:
        """
        Inspect `series` and return a DetectionResult.
        Must never raise — catch internal errors and return empty result.
        """