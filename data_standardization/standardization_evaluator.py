# standardization_evaluator.py
"""
standardization_evaluator.py
=============================
Evaluates standardization pipeline output against ground truth.

Completely decoupled from DataStandardizingService — it only needs:
  - The original dataframe (to know which values were actually observed)
  - A ResultCollector (to read accepted mappings)
  - A ground truth dict

Why separate?
-------------
Evaluation is a benchmarking/testing concern, not a runtime concern.
The service runs in production; the evaluator runs in notebooks,
test suites, and ablation studies. They should not be coupled.

Usage
-----
    evaluator = StandardizationEvaluator(
        original_df=original_df,
        collector=collector,
    )
    report = evaluator.evaluate(ground_truth)
    evaluator.print_report(report)
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field

from .result_collector import ResultCollector

import pandas as pd


# ------------------------------------------------------------------
# Value objects
# ------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnEvaluation:
    """Precision / recall / F1 for a single column."""
    column: str
    precision: float | None
    recall: float | None
    f1: float
    tp: int
    fp: int
    fn: int
    total_evaluated: int
    fallback_rate: float


@dataclass(frozen=True)
class EvaluationReport:
    """
    Full evaluation report across all columns.

    Attributes
    ----------
    per_column  : One ColumnEvaluation per evaluated column.
    overall     : Micro-averaged metrics across all columns.
    """
    per_column: dict[str, ColumnEvaluation]
    overall_precision: float | None
    overall_recall: float | None
    overall_f1: float
    overall_fallback_rate: float
    total_evaluated: int

    def to_dict(self) -> dict:
        """
        Export in the same shape as the old evaluate() return value.
        Keeps notebooks and test suites unaffected.
        """
        def _fmt(v: float | None) -> float | str:
            return round(v, 3) if v is not None else "N/A"

        return {
            "per_column": {
                col: {
                    "precision": _fmt(e.precision),
                    "recall": _fmt(e.recall),
                    "f1": round(e.f1, 3),
                    "tp": e.tp,
                    "fp": e.fp,
                    "fn": e.fn,
                    "total_evaluated": e.total_evaluated,
                    "fallback_rate": round(e.fallback_rate, 3),
                }
                for col, e in self.per_column.items()
            },
            "overall": {
                "precision": _fmt(self.overall_precision),
                "recall": _fmt(self.overall_recall),
                "f1": round(self.overall_f1, 3),
                "fallback_rate": round(self.overall_fallback_rate, 3),
                "total_evaluated": self.total_evaluated,
            },
        }


# ------------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------------

class StandardizationEvaluator:
    """
    Compares pipeline output against a ground-truth mapping.

    Parameters
    ----------
    original_df : The dataframe BEFORE standardization was applied.
                  Used to filter ground truth to only observed values.
    collector   : ResultCollector from the pipeline run being evaluated.
    """

    def __init__(
        self,
        original_df: pd.DataFrame,
        collector: ResultCollector,
    ) -> None:
        self._original_df = original_df
        self._collector = collector

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        ground_truth: dict[str, dict[str, str]],
    ) -> EvaluationReport:
        """
        Evaluate pipeline output against ground truth.

        Parameters
        ----------
        ground_truth : { column_name: { dirty_value: expected_canonical } }

        Returns
        -------
        EvaluationReport with per-column and overall metrics.
        """
        per_column: dict[str, ColumnEvaluation] = {}
        totals = {"tp": 0, "fp": 0, "fn": 0, "total": 0}

        for column, gt_map in ground_truth.items():
            evaluation = self._evaluate_column(column, gt_map)
            per_column[column] = evaluation
            totals["tp"] += evaluation.tp
            totals["fp"] += evaluation.fp
            totals["fn"] += evaluation.fn
            totals["total"] += evaluation.total_evaluated

        overall_precision, overall_recall, overall_f1 = self._metrics(
            totals["tp"], totals["fp"], totals["fn"]
        )

        return EvaluationReport(
            per_column=per_column,
            overall_precision=overall_precision,
            overall_recall=overall_recall,
            overall_f1=overall_f1,
            overall_fallback_rate=(
                round(totals["fn"] / totals["total"], 3)
                if totals["total"] else 0.0
            ),
            total_evaluated=totals["total"],
        )

    def print_report(self, report: EvaluationReport) -> None:
        """Pretty-print an EvaluationReport to stdout."""
        print("\n=== EVALUATION REPORT ===")
        print(json.dumps(report.to_dict(), indent=2))

    # ------------------------------------------------------------------
    # Internal — per-column evaluation
    # ------------------------------------------------------------------

    def _evaluate_column(
        self,
        column: str,
        gt_map: dict[str, str],
    ) -> ColumnEvaluation:
        applied_mapping = self._collector.accepted_changes(column)
        observed = self._observed_values(column)

        # Only evaluate values actually seen in the original dataframe
        filtered = {
            orig: expected
            for orig, expected in gt_map.items()
            if not observed or orig in observed
        }

        tp = fp = fn = 0
        for original, expected in filtered.items():
            predicted = applied_mapping.get(original, original)
            correct = self._normalize(predicted) == self._normalize(expected)
            changed = original in applied_mapping

            if changed and correct:
                tp += 1
            elif changed and not correct:
                fp += 1
            elif not changed and self._normalize(expected) != self._normalize(original):
                fn += 1

        precision, recall, f1 = self._metrics(tp, fp, fn)

        return ColumnEvaluation(
            column=column,
            precision=precision,
            recall=recall,
            f1=f1,
            tp=tp,
            fp=fp,
            fn=fn,
            total_evaluated=len(filtered),
            fallback_rate=round(fn / len(filtered), 3) if filtered else 0.0,
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _observed_values(self, column: str) -> set[str]:
        if column not in self._original_df.columns:
            return set()
        return set(
            self._original_df[column]
            .dropna()
            .astype(str)
            .str.strip()
            .unique()
        )

    @staticmethod
    def _normalize(value: str) -> str:
        """
        Case-fold, unicode-normalize, and strip whitespace for comparison.
        Matches the original evaluate() normalization exactly.
        """
        s = unicodedata.normalize("NFKC", value).lower().strip()
        return re.sub(r"\s+", "", s)

    @staticmethod
    def _metrics(
        tp: int,
        fp: int,
        fn: int,
    ) -> tuple[float | None, float | None, float]:
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None
            and recall is not None
            and (precision + recall) > 0
            else 0.0
        )
        return precision, recall, f1