"""
Raw minimal baseline for benchmark datasets.

This script evaluates the original raw CSV files without AutoPrepAI or
fixed-rule cleaning. "Minimal preprocessing" means only the transformations
required for RandomForestClassifier to accept the data safely:

- normalize common missing-value markers;
- drop rows only when the target itself is missing;
- impute missing feature values;
- one-hot encode categorical feature columns.

It does not remove duplicates, remove outliers, select features, engineer
features, or apply adaptive cleaning logic.

Example:
    python run_raw_minimal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from run_fixed_rule import DATASETS, encode_target, normalize_missing_markers, resolve_target


RESULTS_PATH = Path("eval/raw_minimal_results.json")
ALL_RESULTS_PATH = Path("eval/raw_minimal_all_results.json")
ALL_RESULTS_CSV_PATH = Path("eval/raw_minimal_all_results.csv")
EXPERIMENT = "raw_minimal"


def make_one_hot_encoder() -> OneHotEncoder:
    """Create a version-compatible one-hot encoder."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def feature_count(preprocessor: ColumnTransformer, numeric_features: list[str]) -> int:
    """Return transformed feature count after train-fitted preprocessing."""
    count = len(numeric_features)
    if "categorical" in preprocessor.named_transformers_:
        encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
        count += len(encoder.get_feature_names_out())
    return count


def evaluate_raw_minimal(X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, float], int]:
    """Split, fit train-only preprocessing, train RandomForest, and score.

    The split and RandomForest configuration match the existing evaluation
    scripts. Feature preprocessing is fit only on X_train to avoid train/test
    leakage from imputation or category discovery.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y if y.nunique() > 1 else None,
    )

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = [col for col in X.columns if col not in categorical_features]

    transformers = []
    if numeric_features:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric_features))
    if categorical_features:
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42)),
        ]
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    fitted_preprocessor = model.named_steps["preprocessor"]
    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0),
    }
    return metrics, feature_count(fitted_preprocessor, numeric_features)


def evaluate_dataset(dataset: str, input_path: str, target: str | None = None) -> dict[str, Any]:
    df = pd.read_csv(input_path)
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    target = resolve_target(df, dataset, target)
    rows_before = len(df)

    df = normalize_missing_markers(df)
    df = df.replace({pd.NA: np.nan})
    target_missing = df[target].isna()
    if target_missing.any():
        df = df.loc[~target_missing].reset_index(drop=True)

    y = encode_target(df[target])
    X = df.drop(columns=[target])
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    metrics, features_selected = evaluate_raw_minimal(X, y)
    return {
        "dataset": dataset,
        "experiment": EXPERIMENT,
        **metrics,
        "rows_before": rows_before,
        "rows_after": int(len(df)),
        "features_selected": int(features_selected),
        "categorical_features_encoded": categorical_features,
        "duplicates_removed": 0,
        "outliers_removed": 0,
        "input": str(Path(input_path)),
        "target": target,
    }


def default_runs() -> list[dict[str, str | None]]:
    runs: list[dict[str, str | None]] = []
    for dataset, config in DATASETS.items():
        path = str(config["path"])
        if Path(path).exists():
            runs.append({"dataset": dataset, "path": path, "target": None})
        else:
            print(f"Skipping {dataset}: file not found at {path}")
    return runs


def collect_runs() -> list[dict[str, str | None]]:
    if not sys.stdin.isatty():
        return default_runs()

    runs: list[dict[str, str | None]] = []
    for dataset, config in DATASETS.items():
        default_path = str(config["path"])
        path = input(f"Enter raw CSV path for {dataset} [{default_path}, or 'skip']: ").strip()
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


def main() -> None:
    runs = collect_runs() or default_runs()
    if not runs:
        raise ValueError("No dataset files found in datasets/.")

    results = []
    failures = []
    for run in runs:
        dataset = str(run["dataset"])
        print(f"\nRunning raw minimal baseline on {dataset}...")
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
        raise ValueError("No datasets could be evaluated. Check that each CSV has a target column.")

    RESULTS_PATH.write_text(json.dumps(results[-1], indent=2), encoding="utf-8")
    ALL_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(results).to_csv(ALL_RESULTS_CSV_PATH, index=False)
    print(f"\nSaved last result to {RESULTS_PATH.resolve()}")
    print(f"Saved all raw-minimal results to {ALL_RESULTS_PATH.resolve()} and {ALL_RESULTS_CSV_PATH.resolve()}")
    if failures:
        print("\nDatasets skipped:")
        print(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
