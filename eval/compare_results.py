"""
Compare raw-minimal, fixed-rule, and AutoPrepAI evaluation results.

Example:
    python compare_results.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


OUTPUT_CSV_PATH = Path("eval/fixed_rule_vs_autoprepai_comparison.csv")
OUTPUT_JSON_PATH = Path("eval/fixed_rule_vs_autoprepai_comparison.json")

METRICS = [
    "accuracy",
    "precision_weighted",
    "recall_weighted",
    "f1_weighted",
    "rows_before",
    "rows_after",
    "features_selected",
    "duplicates_removed",
    "outliers_removed",
]

QUALITY_METRICS = {"accuracy", "precision_weighted", "recall_weighted", "f1_weighted"}


def load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return [data]


def optional_load_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return load_results(path)


def winner(metric: str, values: dict[str, Any]) -> str:
    available = {name: value for name, value in values.items() if not pd.isna(value)}
    if len(available) < 2:
        return "n/a"
    if len(set(available.values())) == 1:
        return "tie"
    if metric in QUALITY_METRICS:
        return max(available, key=available.get)
    return "n/a"


def build_comparison(
    fixed_results: list[dict[str, Any]],
    autoprep_results: list[dict[str, Any]],
    raw_results: list[dict[str, Any]] | None = None,
) -> pd.DataFrame:
    raw_results = raw_results or []
    raw_by_dataset = {item.get("dataset", "single"): item for item in raw_results}
    fixed_by_dataset = {item.get("dataset", "single"): item for item in fixed_results}
    autoprep_by_dataset = {item.get("dataset", "single"): item for item in autoprep_results}
    datasets = sorted(set(raw_by_dataset) | set(fixed_by_dataset) | set(autoprep_by_dataset))

    rows = []
    for dataset in datasets:
        raw = raw_by_dataset.get(dataset, {})
        fixed = fixed_by_dataset.get(dataset, {})
        autoprep = autoprep_by_dataset.get(dataset, {})
        for metric in METRICS:
            raw_value = raw.get(metric)
            fixed_value = fixed.get(metric)
            autoprep_value = autoprep.get(metric)
            row = {
                "dataset": dataset,
                "metric": metric,
                "raw_minimal": raw_value,
                "fixed_rule": fixed_value,
                "autoprepai": autoprep_value,
                "winner": winner(
                    metric,
                    {
                        "raw_minimal": raw_value,
                        "fixed_rule": fixed_value,
                        "AutoPrepAI": autoprep_value,
                    },
                ),
            }
            if metric in QUALITY_METRICS and fixed_value is not None and autoprep_value is not None:
                row["autoprepai_minus_fixed_rule"] = autoprep_value - fixed_value
            else:
                row["autoprepai_minus_fixed_rule"] = None
            if metric in QUALITY_METRICS and raw_value is not None and autoprep_value is not None:
                row["autoprepai_minus_raw_minimal"] = autoprep_value - raw_value
            else:
                row["autoprepai_minus_raw_minimal"] = None
            if metric in QUALITY_METRICS and raw_value is not None and fixed_value is not None:
                row["fixed_rule_minus_raw_minimal"] = fixed_value - raw_value
            else:
                row["fixed_rule_minus_raw_minimal"] = None
            rows.append(row)
    return pd.DataFrame(rows)


def print_table(df: pd.DataFrame) -> None:
    headers = df.columns.tolist()
    values = df.astype(object).where(pd.notna(df), "").values.tolist()
    widths = [
        max(len(str(row[i])) for row in [headers, *values])
        for i in range(len(headers))
    ]

    def fmt(row: list[Any]) -> str:
        return " | ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))

    print(fmt(headers))
    print("-+-".join("-" * width for width in widths))
    for row in values:
        print(fmt(row))


def unlocked_path(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")


def main() -> None:
    if sys.stdin.isatty():
        raw_path = input("Enter raw-minimal all-results JSON [eval/raw_minimal_all_results.json, or 'skip']: ").strip()
        raw_path = "" if raw_path.lower() == "skip" else raw_path or "eval/raw_minimal_all_results.json"
        fixed_path = input("Enter fixed-rule all-results JSON [eval/fixed_rule_all_results.json]: ").strip()
        fixed_path = fixed_path or "eval/fixed_rule_all_results.json"
        autoprep_path = input("Enter AutoPrepAI all-results JSON [eval/autoprepai_all_results.json]: ").strip()
        autoprep_path = autoprep_path or "eval/autoprepai_all_results.json"
    else:
        raw_path = "eval/raw_minimal_all_results.json"
        fixed_path = "eval/fixed_rule_all_results.json"
        autoprep_path = "eval/autoprepai_all_results.json"

    raw_results = optional_load_results(Path(raw_path)) if raw_path else []
    fixed_results = optional_load_results(Path(fixed_path)) if fixed_path else []
    autoprep_results = optional_load_results(Path(autoprep_path)) if autoprep_path else []
    if not raw_results and not fixed_results and not autoprep_results:
        raise FileNotFoundError("No result files found to compare.")
    comparison = build_comparison(fixed_results, autoprep_results, raw_results)

    print_table(comparison)
    csv_path = OUTPUT_CSV_PATH
    json_path = OUTPUT_JSON_PATH
    try:
        comparison.to_csv(csv_path, index=False)
    except PermissionError:
        csv_path = unlocked_path(OUTPUT_CSV_PATH)
        comparison.to_csv(csv_path, index=False)

    try:
        json_path.write_text(comparison.to_json(orient="records", indent=2), encoding="utf-8")
    except PermissionError:
        json_path = unlocked_path(OUTPUT_JSON_PATH)
        json_path.write_text(comparison.to_json(orient="records", indent=2), encoding="utf-8")

    print(f"\nSaved comparison to {csv_path.resolve()} and {json_path.resolve()}")


if __name__ == "__main__":
    main()
