import time
import warnings
import pandas as pd
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score, roc_auc_score

warnings.filterwarnings("ignore")



@dataclass
class BenchmarkResult:
    strategy_name: str

    rows_before: int
    rows_after: int
    removed_rows: int
    removal_rate: float

    # cleaned-data scores
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    execution_time: float


def _encode_target(y: pd.Series) -> pd.Series:
    """Label-encode string targets so sklearn is happy."""
    if y.dtype == object or str(y.dtype) == "category":
        le = LabelEncoder()
        return pd.Series(le.fit_transform(y), index=y.index, name=y.name)
    return y


def _is_binary(y: pd.Series) -> bool:
    return y.nunique() == 2


def _safe_roc_auc(model, X_test, y_test) -> float:
    """ROC-AUC only makes sense for binary targets."""
    try:
        if _is_binary(y_test):
            y_prob = model.predict_proba(X_test)[:, 1]
            return roc_auc_score(y_test, y_prob)
        else:
            # multi-class OvR
            y_prob = model.predict_proba(X_test)
            return roc_auc_score(
                y_test, y_prob,
                multi_class="ovr",
                average="weighted"
            )
    except Exception:
        return float("nan")



def _cv_evaluate(
    df: pd.DataFrame,
    target_column: str,
    random_state: int = 42,
    n_splits: int = 5
) -> Dict[str, float]:


    X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])
    X = X.fillna(X.median())

    y = _encode_target(df[target_column])

    valid = y.notna()
    X, y = X[valid], y[valid]

    if y.nunique() < 2:
        return {k: float("nan") for k in
                ["accuracy", "precision", "recall", "f1", "roc_auc"]}

    n_splits = min(n_splits, y.value_counts().min())
    n_splits = max(n_splits, 2)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=random_state,
        n_jobs=-1
    )

    scoring = {
        "accuracy":  "accuracy",
        "precision": "precision_weighted",
        "recall":    "recall_weighted",
        "f1":        "f1_weighted",
    }

    cv_results = cross_validate(
        model, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score="raise"
    )

    # ROC-AUC needs predict_proba — do it manually per fold
    roc_scores = []
    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
        m = RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1
        )
        m.fit(X_tr, y_tr)
        roc_scores.append(_safe_roc_auc(m, X_te, y_te))

    return {
        "accuracy":  float(np.mean(cv_results["test_accuracy"])),
        "precision": float(np.mean(cv_results["test_precision"])),
        "recall":    float(np.mean(cv_results["test_recall"])),
        "f1":        float(np.mean(cv_results["test_f1"])),
        "roc_auc":   float(np.nanmean(roc_scores)),
    }


class OutlierBenchmarkEvaluator:

    def __init__(
        self,
        dataframe: pd.DataFrame,
        target_column: str,
        strategies: Dict[str, Any],
        random_state: int = 42,
        cv_folds: int = 5,
    ):
        self.df = dataframe.copy()
        self.target_column = target_column
        self.strategies = strategies
        self.random_state = random_state
        self.cv_folds = cv_folds

        self.results: List[BenchmarkResult] = []
        self._baseline_metrics: Optional[Dict[str, float]] = None


    def evaluate(self) -> pd.DataFrame:

        print("\nComputing baseline (no outlier removal)…")
        self._baseline_metrics = _cv_evaluate(
            self.df,
            self.target_column,
            self.random_state,
            self.cv_folds
        )
        print(f"  Baseline F1      : {self._baseline_metrics['f1']:.4f}")
        print(f"  Baseline ROC-AUC : {self._baseline_metrics['roc_auc']:.4f}")

        for name, strategy in self.strategies.items():
            print(f"\n{'─' * 60}")
            print(f"Evaluating: {name}")
            print(f"{'─' * 60}")
            result = self._evaluate_single(name, strategy)
            self.results.append(result)

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.results:
            rows.append({
                "Strategy":           r.strategy_name,
                "Rows Before":        r.rows_before,
                "Rows After":         r.rows_after,
                "Removed Rows":       r.removed_rows,
                "Removal Rate":       round(r.removal_rate, 4),
                "Accuracy":           round(r.accuracy,  4),
                "Precision":          round(r.precision, 4),
                "Recall":             round(r.recall,    4),
                "F1":                 round(r.f1,        4),
                "ROC-AUC":            round(r.roc_auc,   4),
                "Execution Time (s)": round(r.execution_time, 4),
            })
        return pd.DataFrame(rows)

    def print_summary(self):
        df = self.to_dataframe()
        print("\n" + "=" * 110)
        print("OUTLIER STRATEGY BENCHMARK RESULTS")
        print("=" * 110)
        print(df.to_string(index=False))
        print("=" * 110)

    def export_csv(self, path: str = "outlier_benchmark_results.csv"):
        self.to_dataframe().to_csv(path, index=False)
        print(f"Results exported to: {path}")

    

    def _evaluate_single(self, name: str, strategy) -> BenchmarkResult:
        start = time.time()
        df_copy = self.df.copy()
        mask = strategy.detect(df_copy)
        cleaned_df = df_copy[mask].copy()
        elapsed = time.time() - start
        rows_before = len(df_copy)
        rows_after  = len(cleaned_df)
        removed     = rows_before - rows_after
        removal_rate = removed / rows_before

        print(f"  Rows before : {rows_before}")
        print(f"  Rows after  : {rows_after}  (removed {removed}, {removal_rate:.1%})")

       
        metrics = _cv_evaluate(
            cleaned_df,
            self.target_column,
            self.random_state,
            self.cv_folds
        )


        return BenchmarkResult(
            strategy_name=name,
            rows_before=rows_before,
            rows_after=rows_after,
            removed_rows=removed,
            removal_rate=removal_rate,
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
            roc_auc=metrics["roc_auc"],
            execution_time=elapsed,
        )