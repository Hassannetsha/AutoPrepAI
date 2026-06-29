"""
data_standardizing_service.py
==============================
Orchestrator for the Categorical Feature Inconsistency Agent pipeline.

Responsibilities (only these)
------------------------------
1. Accept dependencies at construction time.
2. Route each column to the categorical detector.
3. Apply accepted changes to the dataframe.
4. Forward results and logs to ResultCollector.

Everything else lives in a dedicated module:
    Rate limiting       → RateLimiter
    LLM calls/retries   → GroqLLMClient
    Numeric detection   → NumericRuleDetector / NumericLLMDetector (called from main pipeline)
    Categorical detect  → CategoricalClusterDetector
    Result storage      → ResultCollector
    Evaluation          → StandardizationEvaluator (Step 7)
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .categorical_cluster_detector import CategoricalClusterDetector
from .detection_strategy import DetectionContext
from .llm_client import LLMClient
from .result_collector import (
    ColumnStandardizationResult,
    ResultCollector,
    ValidationEvent,
)

try:
    from .validation_layer import ValidationLayer
except ImportError:
    from validation_layer import ValidationLayer


class DataStandardizingService:
    """
    Orchestrates categorical inconsistency detection and standardization.

    Parameters
    ----------
    df                      : Input dataframe. A copy is made immediately —
                              the original is never modified.
    llm_client              : Any LLMClient implementation.
    validation              : ValidationLayer carrying domain rules and whitelists.
    confidence_threshold    : Minimum confidence to accept any mapping.
    similarity_threshold    : Edit-distance threshold for categorical clustering.
    max_unique_values       : Cap on unique values sent to the clusterer per column.
    min_unique_for_auto     : Low-cardinality skip threshold for categorical columns.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        llm_client: LLMClient,
        validation: Optional[ValidationLayer] = None,
        confidence_threshold: float = 0.7,
        similarity_threshold: float = 0.35,
        max_unique_values: int = 500,
        min_unique_for_auto: int = 20,
    ) -> None:
        self.df = df.copy().reset_index(drop=True)
        self.original_df = df.copy().reset_index(drop=True)

        self._llm_client = llm_client
        self._validation = validation or ValidationLayer()
        self._confidence_threshold = confidence_threshold
        self._collector = ResultCollector()

        self._categorical_detector = CategoricalClusterDetector(
            llm_client=llm_client,
            validation=self._validation,
            confidence_threshold=confidence_threshold,
            similarity_threshold=similarity_threshold,
            max_unique_values=max_unique_values,
            min_unique_for_auto=min_unique_for_auto,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def results(self) -> dict:
        """Backward-compatible export for callers that used self.results."""
        return self._collector.to_dict()

    def standardize(
        self,
        columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
    ) -> pd.DataFrame:  # <-- Added return type
        """
        Run the full pipeline over the dataframe.
        Returns the cleaned DataFrame.
        """
        target = columns or list(self.df.columns)

        cat_cols = (
            categorical_columns if categorical_columns is not None
            else [c for c in target if not pd.api.types.is_numeric_dtype(self.df[c])]
        )

        print(f"Standardizing {len(cat_cols)} categorical columns.")

        for col in cat_cols:
            print(f"\n[categorical] {col}")
            self._process_categorical(col)

        return self.df 

    def standardize_column(self, column: str) -> ColumnStandardizationResult:
        """Run the categorical pipeline on a single column."""
        return self._process_categorical(column)

    # ------------------------------------------------------------------
    # Internal — categorical
    # ------------------------------------------------------------------

    def _process_categorical(self, column: str) -> ColumnStandardizationResult:
        """Detect variants, apply mapping, record result."""
        series = self.df[column].dropna().astype(str)
        context = self._make_context(column)

        detection = self._categorical_detector.detect(series, context)

        if not detection.issues:
            result = ColumnStandardizationResult(
                column=column,
                skipped=not bool(series.nunique()),
            )
            self._collector.set_column_result(result)
            return result

        # Build mapping from issues — variant -> canonical
        mapping = {
            issue.value: issue.canonical
            for issue in detection.issues
            if issue.canonical is not None
        }

        # ── BRUTE-FORCE REPLACEMENT ──
        # Bypasses .replace() and .map() entirely to avoid any Pandas 
        # dtype/category/string wrapper silent failures.
        for variant, canonical in mapping.items():
            # .strip() ensures hidden spaces don't break the match
            mask = self.df[column].astype(str).str.strip() == variant.strip()
            self.df.loc[mask, column] = canonical

        # Forward validation log from resolver
        self._collector.log_many([
            ValidationEvent(
                stage=entry.get("stage", "cluster"),
                column=column,
                action="accepted" if entry.get("accepted") else "rejected",
                original_value=entry.get("original"),
                mapped_value=entry.get("canonical"),
                confidence=entry.get("confidence"),
                reason=entry.get("reason"),
                accepted=entry.get("accepted"),
            )
            for entry in detection.log
        ])

        result = ColumnStandardizationResult(
            column=column,
            accepted_changes=mapping,
            total_unique_checked=int(series.nunique()),
            total_accepted=int(len(mapping)),
            total_rejected=int(series.nunique() - len(mapping)),
        )
        self._collector.set_column_result(result)

        print(f"  accepted={len(mapping)}")
        return result

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _make_context(self, column: str) -> DetectionContext:
        return DetectionContext(
            column=column,
            allowed_values=self._get_allowed_values(column),
            rules=self._validation.get_rules(column),
        )

    def _get_allowed_values(self, column: str) -> Optional[list[str]]:
        rules = self._validation.get_rules(column)
        values = rules.get("allowed_values", [])
        
        if not values:
            return None  # triggers discovery mode in resolver
    
        return sorted(str(v) for v in values)