"""
Run realistic gate-logging examples on data_standardization/adult.csv.

This script uses:
- the real Adult dataset stored in this repository;
- the real Groq client and model;
- the real DataStandardizingService confidence and validation gates;
- validation vocabularies derived from the real dataset.

Because the LLM response is live, a given run may or may not produce both
rejection types. The script reports the first observed rejection for each gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from groq import Groq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_key_manager import get_key_manager
from data_standardization.data_standardizing_service import DataStandardizingService
from data_standardization.validation_layer import ValidationLayer


DEFAULT_COLUMNS = ["workclass", "occupation", "native.country", "income", "sex"]


def load_dataset() -> pd.DataFrame:
    dataset_path = Path(__file__).with_name("adult.csv")
    df = pd.read_csv(dataset_path)
    print(f"Dataset: {dataset_path}")
    print(f"Shape: {df.shape}")
    return df


def unique_values_to_check(df: pd.DataFrame, columns: list[str], max_values: int) -> pd.DataFrame:
    rows = {}
    for column in columns:
        if column not in df.columns:
            continue

        values = df[column].dropna().astype(str).str.strip()
        # Prefer suspicious values first, then include a small sample of ordinary values.
        suspicious = values[
            values.isin(["?", "??", "unknown", "Unknown", "N/A", "NA", ""])
        ].drop_duplicates()
        ordinary = values[~values.isin(suspicious)].drop_duplicates()
        selected = pd.concat([suspicious, ordinary]).head(max_values).tolist()
        if selected:
            rows[column] = selected

    max_len = max((len(v) for v in rows.values()), default=0)
    padded = {
        column: values + [pd.NA] * (max_len - len(values))
        for column, values in rows.items()
    }
    return pd.DataFrame(padded)


def realistic_corrupted_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a tiny dirty-data frame from real Adult values.

    The values are intentionally placed into plausible-but-problematic fields to
    exercise the real gates with real LLM proposals: for example, an occupation
    label appearing in the sex column is a realistic column-contamination error.
    """
    occupation_value = (
        df["occupation"].dropna().astype(str).str.strip().loc[lambda s: s != "?"].iloc[0]
    )
    country_value = (
        df["native.country"].dropna().astype(str).str.strip().loc[lambda s: s != "?"].iloc[0]
    )

    return pd.DataFrame(
        {
            "sex": ["?", occupation_value],
            "income": ["?", country_value],
            "occupation": ["?", "not listed"],
            "native.country": ["?", "not listed"],
        }
    )


def build_validation_layer(df: pd.DataFrame, columns: list[str]) -> ValidationLayer:
    validation = ValidationLayer()
    for column in columns:
        if column not in df.columns:
            continue

        values = df[column].dropna().astype(str).str.strip()
        allowed = set(values[~values.isin({"?", "??", "unknown", "Unknown", "N/A", "NA", ""})].unique())
        if allowed:
            validation.register(column, allowed_values=allowed)

    return validation


def make_service(
    df: pd.DataFrame,
    *,
    client: Groq,
    model: str,
    threshold: float,
    validation_layer: ValidationLayer | None = None,
) -> DataStandardizingService:
    return DataStandardizingService(
        df=df,
        client=client,
        model=model,
        confidence_threshold=threshold,
        validation_layer=validation_layer,
        requests_per_minute=8,
        tokens_per_minute=12000,
    )


def first_rejection(logs: list[dict], marker: str) -> dict | None:
    for entry in logs:
        if not entry.get("accepted") and marker in entry.get("fallback_reason", ""):
            return entry
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama-3.3-70b-versatile")
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--max-values", type=int, default=8)
    parser.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    args = parser.parse_args()

    df = load_dataset()
    test_df = unique_values_to_check(df, args.columns, args.max_values)
    if test_df.empty:
        raise RuntimeError("No values selected for testing.")

    print(f"Columns tested: {list(test_df.columns)}")
    print("Values tested:")
    print(test_df.to_string(index=False))
    print()

    key_manager = get_key_manager()
    client = Groq(api_key=key_manager.get_current_key())

    print("=== Scenario 1: confidence gate only, no validation rules ===")
    confidence_service = make_service(
        test_df,
        client=client,
        model=args.model,
        threshold=args.threshold,
    )
    confidence_service.apply_llm_normalization(columns=list(test_df.columns))
    confidence_rejection = first_rejection(
        confidence_service.results["validation_log"],
        "confidence ",
    )
    if confidence_rejection:
        print(json.dumps(confidence_rejection, indent=2, ensure_ascii=False))
    else:
        print("No confidence-gate rejection observed in this live run.")
    print()

    print("=== Scenario 2: confidence + constraint gates, V from adult.csv ===")
    validation = build_validation_layer(df, list(test_df.columns))
    constraint_service = make_service(
        test_df,
        client=client,
        model=args.model,
        threshold=args.threshold,
        validation_layer=validation,
    )
    constraint_service.apply_llm_normalization(columns=list(test_df.columns))
    constraint_rejection = first_rejection(
        constraint_service.results["validation_log"],
        "validation failed:",
    )
    if constraint_rejection:
        print(json.dumps(constraint_rejection, indent=2, ensure_ascii=False))
    else:
        print("No constraint-gate rejection observed in this live run.")
    print()

    print("=== Scenario 3: realistic corrupted cells from adult.csv values ===")
    corrupted_df = realistic_corrupted_values(df)
    print(corrupted_df.to_string(index=False))
    corrupted_validation = build_validation_layer(df, list(corrupted_df.columns))
    corrupted_service = make_service(
        corrupted_df,
        client=client,
        model=args.model,
        threshold=args.threshold,
        validation_layer=corrupted_validation,
    )
    corrupted_service.apply_llm_normalization(columns=list(corrupted_df.columns))

    observed_rejections = [
        entry
        for entry in corrupted_service.results["validation_log"]
        if not entry.get("accepted")
    ]
    if observed_rejections:
        print(json.dumps(observed_rejections, indent=2, ensure_ascii=False))
    else:
        print("No rejection observed for the corrupted-cell scenario in this live run.")

if __name__ == "__main__":
    main()