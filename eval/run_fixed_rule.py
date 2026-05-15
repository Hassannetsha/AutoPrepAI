"""
Fixed-rule preprocessing baseline for one or more benchmark datasets.

Example:
    python run_fixed_rule.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


RESULTS_PATH = Path("eval/fixed_rule_results.json")
ALL_RESULTS_PATH = Path("eval/fixed_rule_all_results.json")
ALL_RESULTS_CSV_PATH = Path("eval/fixed_rule_all_results.csv")

DATASETS: dict[str, dict[str, Any]] = {
    "adult": {"path": "datasets/adult.csv", "targets": ["income", "class"]},
    # "titanic": {"path": "datasets/titanic.csv", "targets": ["Survived", "survived"]},
    # "german_credit": {
    #     "path": "datasets/german_credit.csv",
    #     "targets": ["class", "target", "credit_risk", "Risk", "risk"],
    # },
    "diabetes": {"path": "datasets/diabetic_data.csv", "targets": ["readmitted"]},
    # "breast_cancer": {"path": "datasets/breast_cancer.csv", "targets": ["diagnosis", "target", "class"]},
    # "heart": {"path": "datasets/heart.csv", "targets": ["target"]},
    # "horse" : {"path": "datasets/horse.csv", "targets": ["outcome"]},
}


def normalize_missing_markers(df: pd.DataFrame) -> pd.DataFrame:
    """Treat common text missing markers and blank strings as missing values."""
    cleaned = df.copy()
    cleaned = cleaned.replace(["?", " ?", "? ", " ? ", "", "nan", "NaN", "NA", "N/A", "null"], pd.NA)
    for col in cleaned.select_dtypes(include=["object", "category"]).columns:
        cleaned[col] = cleaned[col].map(lambda value: pd.NA if pd.isna(value) else str(value).strip())
    return cleaned


def impute_missing_values(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, int]:
    """Fixed-rule missing handling: mean for numeric features, mode for categorical features.

    The target is never imputed. Rows with missing target values are removed,
    because inventing labels would create target leakage.
    """
    imputed = df.copy()
    rows_before_target_drop = len(imputed)
    imputed = imputed.dropna(subset=[target]).reset_index(drop=True)
    rows_removed_missing_target = rows_before_target_drop - len(imputed)

    feature_cols = [col for col in imputed.columns if col != target]
    numeric_cols = imputed[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [col for col in feature_cols if col not in numeric_cols]

    for col in numeric_cols:
        mean_value = imputed[col].mean()
        if pd.isna(mean_value):
            mean_value = 0
        imputed[col] = imputed[col].fillna(mean_value)

    for col in categorical_cols:
        mode = imputed[col].mode(dropna=True)
        fill_value = mode.iloc[0] if not mode.empty else "missing"
        imputed[col] = imputed[col].fillna(fill_value)

    return imputed, rows_removed_missing_target


def resolve_target(df: pd.DataFrame, dataset: str, target: str | None) -> str:
    if target and target in df.columns:
        return target
    if target:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    for candidate in DATASETS[dataset]["targets"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"No default target found for {dataset}. Tried {DATASETS[dataset]['targets']}. "
        f"Available columns: {list(df.columns)}"
    )


def encode_target(y: pd.Series) -> pd.Series:
    """Map common binary labels when present; otherwise factorize labels for reuse."""
    cleaned = y.astype(str).str.replace(".", "", regex=False).str.strip()
    binary_maps = [
        {"<=50K": 0, ">50K": 1},
        {"no": 0, "yes": 1},
        {"No": 0, "Yes": 1},
        {"0": 0, "1": 1},
        {"B": 0, "M": 1},
    ]
    values = set(cleaned.dropna().unique())
    for label_map in binary_maps:
        if values.issubset(label_map):
            return cleaned.map(label_map).astype(int)

    codes, uniques = pd.factorize(cleaned)
    if len(uniques) < 2:
        raise ValueError("Target must contain at least two classes after preprocessing.")
    return pd.Series(codes, index=y.index)


def iqr_outlier_mask(df: pd.DataFrame, numeric_cols: list[str]) -> pd.Series:
    """Flag rows outside 1.5x IQR on any numeric feature column."""
    mask = pd.Series(False, index=df.index)
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors="coerce")
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            continue
        mask |= (values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)
    return mask.fillna(False)


def evaluate(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Split, train RandomForest, and return weighted metrics."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0),
    }


def evaluate_dataset(dataset: str, input_path: str, target: str | None = None) -> dict[str, Any]:
    df = pd.read_csv(input_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    target = resolve_target(df, dataset, target)
    rows_before = len(df)

    df = normalize_missing_markers(df)
    df, rows_removed_missing_target = impute_missing_values(df, target)
    if df.empty:
        raise ValueError(
            "All rows were removed because the target column is missing for every row. "
            "Feature values are imputed, but target labels are never imputed."
        )

    rows_before_duplicates = len(df)
    df = df.drop_duplicates(keep="first").reset_index(drop=True)
    duplicates_removed = rows_before_duplicates - len(df)

    numeric_feature_cols = [
        col for col in df.select_dtypes(include=["number"]).columns.tolist()
        if col != target
    ]
    outlier_mask = iqr_outlier_mask(df, numeric_feature_cols)
    outliers_removed = int(outlier_mask.sum())
    df = df.loc[~outlier_mask].reset_index(drop=True)

    y = encode_target(df[target])
    X = df.drop(columns=[target])
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    X = pd.get_dummies(X, dummy_na=False).astype(float)

    metrics = evaluate(X, y)
    return {
        "dataset": dataset,
        **metrics,
        "rows_before": rows_before,
        "rows_after": int(len(df)),
        "features_selected": int(X.shape[1]),
        "categorical_features_encoded": categorical_features,
        "duplicates_removed": int(duplicates_removed),
        "outliers_removed": int(outliers_removed),
        "missing_target_rows_removed": int(rows_removed_missing_target),
        "input": str(Path(input_path)),
        "target": target,
    }


def collect_runs() -> list[dict[str, str | None]]:
    runs: list[dict[str, str | None]] = []
    if not sys.stdin.isatty():
        return default_runs()

    for dataset, config in DATASETS.items():
        default_path = str(config["path"])
        path = input(f"Enter raw/corrupted CSV path for {dataset} [{default_path}, or 'skip']: ").strip()
        if path.lower() == "skip":
            continue
        path = path or default_path
        if not Path(path).exists():
            print(f"Skipping {dataset}: file not found at {path}")
            continue
        default_targets = ", ".join(config["targets"])
        target = input(f"Enter target column for {dataset} [auto: {default_targets}]: ").strip() or None
        runs.append({"dataset": dataset, "path": path, "target": target})
    return runs


def default_runs() -> list[dict[str, str | None]]:
    runs: list[dict[str, str | None]] = []
    for dataset, config in DATASETS.items():
        path = str(config["path"])
        if Path(path).exists():
            runs.append({"dataset": dataset, "path": path, "target": None})
        else:
            print(f"Skipping {dataset}: file not found at {path}")
    return runs


def main() -> None:
    runs = collect_runs()
    if not runs:
        runs = default_runs()
    if not runs:
        raise ValueError("No dataset files found in datasets/.")

    results = []
    failures = []
    for run in runs:
        dataset = str(run["dataset"])
        print(f"\nRunning fixed-rule baseline on {dataset}...")
        try:
            result = evaluate_dataset(dataset, str(run["path"]), run["target"])
        except ValueError as exc:
            failures.append({"dataset": dataset, "error": str(exc), "input": str(run["path"])})
            print(f"Skipping {dataset}: {exc}")
            continue
        else:
            results.append(result)
            print(json.dumps(result, indent=2))

    if not results:
        raise ValueError(
            "No datasets could be evaluated. See skipped dataset errors above; "
            "the cause may be missing targets, all rows removed by preprocessing, or a single-class target."
        )

    RESULTS_PATH.write_text(json.dumps(results[-1], indent=2), encoding="utf-8")
    ALL_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(ALL_RESULTS_CSV_PATH, index=False)
    print(f"\nSaved last result to {RESULTS_PATH.resolve()}")
    print(f"Saved all fixed-rule results to {ALL_RESULTS_PATH.resolve()} and {ALL_RESULTS_CSV_PATH.resolve()}")
    if failures:
        print("\nDatasets skipped:")
        print(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
