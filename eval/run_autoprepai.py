"""
Evaluate AutoPrepAI outputs for one or more benchmark datasets.

Example:
    python run_autoprepai.py
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


RESULTS_PATH = Path("eval/autoprepai_results.json")
ALL_RESULTS_PATH = Path("eval/autoprepai_all_results.json")
ALL_RESULTS_CSV_PATH = Path("eval/autoprepai_all_results.csv")

DATASETS: dict[str, dict[str, Any]] = {
    "adult": {"path": "cleaned_datasets/adult.csv", "targets": ["income", "class"]},
    # "titanic": {"path": "cleaned_datasets/titanic.csv", "targets": ["Survived", "survived"]},
    # "german_credit": {
    #     "path": "cleaned_datasets/german_credit.csv",
    #     "targets": ["class", "target", "credit_risk", "Risk", "risk"],
    # },
    "diabetes": {"path": "cleaned_datasets/diabetic_data.csv", "targets": ["readmitted"]},
    # "breast_cancer": {"path": "cleaned_datasets/breast_cancer.csv", "targets": ["diagnosis", "target", "class"]},
    # "heart": {"path": "cleaned_datasets/heart.csv", "targets": ["target"]},
    # "horse" : {"path": "cleaned_datasets/horse.csv", "targets": ["outcome"]},
}


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
    ]
    values = set(cleaned.dropna().unique())
    for label_map in binary_maps:
        if values.issubset(label_map):
            return cleaned.map(label_map).astype(int)

    codes, uniques = pd.factorize(cleaned)
    if len(uniques) < 2:
        raise ValueError("Target must contain at least two classes.")
    return pd.Series(codes, index=y.index)


def evaluate(X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
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

    y = encode_target(df[target])
    X = df.drop(columns=[target])

    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    if categorical_features:
        X = pd.get_dummies(X, columns=categorical_features, dummy_na=False)

    non_numeric = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()
    if non_numeric:
        raise ValueError(f"Unable to convert feature columns to numeric: {non_numeric}")
    X = X.astype(float)

    metrics = evaluate(X, y)
    return {
        "dataset": dataset,
        **metrics,
        "rows_before": int(len(df)),
        "rows_after": int(len(df)),
        "features_selected": int(X.shape[1]),
        "categorical_features_encoded": categorical_features,
        "duplicates_removed": 0,
        "outliers_removed": 0,
        "input": str(Path(input_path)),
        "target": target,
    }


def collect_runs() -> list[dict[str, str | None]]:
    runs: list[dict[str, str | None]] = []
    if not sys.stdin.isatty():
        return default_runs()

    for dataset, config in DATASETS.items():
        default_path = str(config["path"])
        path = input(f"Enter AutoPrepAI CSV path for {dataset} [{default_path}, or 'skip']: ").strip()
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
        print(f"\nRunning AutoPrepAI evaluation on {dataset}...")
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
    print(f"Saved all AutoPrepAI results to {ALL_RESULTS_PATH.resolve()} and {ALL_RESULTS_CSV_PATH.resolve()}")
    if failures:
        print("\nDatasets skipped:")
        print(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
