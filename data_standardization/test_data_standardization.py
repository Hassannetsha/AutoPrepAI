"""
test_standardizing.py
---------------------
Smoke-test for DataStandardizingService with UCI Adult dataset
(real-world dirty data: income, gender, occupation).
"""

import os
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np

from groq import Groq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_key_manager import get_key_manager
from data_standardization.data_standardizing_service import DataStandardizingService
from data_standardization.validation_layer import ValidationLayer


# ── 1. Load dataset ───────────────────────────────────────────────────────────

def load_uci_adult_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"
    ]

    print("Downloading UCI Adult dataset...")

    data = pd.read_csv(
        url,
        header=None,
        names=columns,
        na_values=[" ?", "?"]
    )

    print(f"Loaded {len(data)} rows. Introducing dirty data...")

    df = data.sample(n=1000, random_state=42).reset_index(drop=True)

    np.random.seed(42)
    n = len(df)

    # ── SEX (FIXED probabilities issue safety) ───────────────────────────────
    dirty_genders = ["male", "Male", "MALE", "M",
                     "femaile", "FEMALE", "F", "female", "??", "unknown"]

    p = np.array([0.2, 0.15, 0.1, 0.08, 0.05, 0.15, 0.07, 0.2, 0.03, 0.02])
    p = p / p.sum()

    df["sex"] = np.random.choice(dirty_genders, size=n, p=p)

    # ── OCCUPATION (FIXED: safer handling of NaNs) ───────────────────────────
    occupations = df["occupation"].dropna().unique()

    dirty_occs = []
    for occ in occupations[:10]:
        if isinstance(occ, str):
            dirty_occs.extend([
                occ.lower(),
                occ.upper(),
                occ[:4],
                occ.replace(" ", ""),
                f"{occ}?",
                f" {occ} "
            ])

    dirty_occs.extend(["??", "N/A", "unknwn", "missng"])

    mask = df["occupation"].notna()
    final_mask = mask & (np.random.rand(n) < 0.15)

    df.loc[final_mask, "occupation"] = np.random.choice(
        dirty_occs,
        size=final_mask.sum(),
        replace=True
    )

    # ── INCOME (FIXED extra probability bug) ────────────────────────────────
    dirty_income = ["<=50K", ">50K", "<=50k", ">50k", "50K",
                    "<=50K.", ">50K,", "low", "high", "??"]

    p = np.array([0.25, 0.25, 0.1, 0.1, 0.05, 0.05, 0.03, 0.03, 0.05, 0.02])
    p = p / p.sum()

    df["income"] = np.random.choice(dirty_income, size=n, p=p)

    # ── COUNTRY (FIXED broadcasting bug) ────────────────────────────────────
    dirty_countries = [
        "United-States", "US", "USA", "u.s.",
        "Cambodia", "Laos", "l.a.o.s.", "?", "unknown",
        "UK", "England"
    ]

    country_mask = df["native_country"].notna()

    rand_mask = country_mask & (np.random.rand(n) < 0.1)

    df.loc[rand_mask, "native_country"] = np.random.choice(
        dirty_countries,
        size=rand_mask.sum(),
        replace=True
    )

    # ── AGE (unchanged) ─────────────────────────────────────────────────────
    age_mask = np.random.rand(n) < 0.02
    df.loc[age_mask, "age"] = np.random.choice([-5, 150, 200, 0, 999], size=age_mask.sum())

    print("=== Sample dirty data ===")
    print(df[["sex", "occupation", "income", "native_country", "age"]].head(15).to_string(index=False))

    return df, occupations[:10].tolist()


def build_ground_truth(seed_occupations: list[str]) -> dict[str, dict[str, str]]:
    occupation_truth = {}
    prefix_to_occ = {}

    for occ in seed_occupations:
        if not isinstance(occ, str):
            continue

        occupation_truth[occ.lower()] = occ
        occupation_truth[occ.upper()] = occ
        occupation_truth[f"{occ}?"] = occ
        occupation_truth[f" {occ} "] = occ

        no_space = occ.replace(" ", "")
        if no_space != occ:
            occupation_truth[no_space] = occ

        prefix = occ[:4]
        prefix_to_occ.setdefault(prefix, set()).add(occ)

    for prefix, mapped in prefix_to_occ.items():
        if len(mapped) == 1:
            occupation_truth[prefix] = next(iter(mapped))

    return {
        "sex": {
            "male": "Male",
            "Male": "Male",
            "MALE": "Male",
            "M": "Male",
            "female": "Female",
            "Female": "Female",
            "FEMALE": "Female",
            "F": "Female",
            "femaile": "Female",
        },
        "income": {
            "<=50k": "<=50K",
            ">50k": ">50K",
            "<=50K.": "<=50K",
            ">50K,": ">50K",
            "low": "<=50K",
            "high": ">50K",
        },
        "occupation": occupation_truth,
        "native_country": {
            "US": "United-States",
            "USA": "United-States",
            "u.s.": "United-States",
            "UK": "England",
        },
    }


df, seed_occupations = load_uci_adult_dataset()


# ── 2. Validation rules ───────────────────────────────────────────────────────

vl = ValidationLayer()

vl.register("sex", allowed_values={"Male", "Female"})
vl.register("income", allowed_values={">50K", "<=50K"})

vl.register(
    "occupation",
    allowed_values={
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Protective-serv", "Priv-house-serv"
    }
)

vl.register(
    "native_country",
    allowed_values={"United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Japan"},
    min_length=2,
    max_length=50
)

vl.register("age", min_value=16, max_value=90)


# ── 3. Ground truth ───────────────────────────────────────────────────────────

GROUND_TRUTH = build_ground_truth(seed_occupations)


# ── 4. Run pipeline ───────────────────────────────────────────────────────────

api_key = os.getenv("GROQ_API_KEY") or get_key_manager().get_current_key()
if not api_key:
    raise EnvironmentError("Set GROQ_API_KEY environment variable.")

client = Groq(api_key=api_key)

DELTAS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# DELTAS = [0.5]
DETECTION_COLUMNS = ["age", "sex", "occupation", "income", "native_country"]
NORMALIZATION_COLUMNS = ["sex", "occupation", "income", "native_country"]


def run_evaluation(delta: float) -> dict:
    print("\n" + "=" * 80)
    print(f"RUNNING UCI ADULT EVALUATION WITH delta={delta}")
    print("=" * 80)

    service = DataStandardizingService(
        df=df.copy(),
        client=client,
        model="openai/gpt-oss-120b",
        confidence_threshold=delta,
        validation_layer=vl,
    )

    print("=== Running Standardization ===")
    service.standardize(
        numeric_columns=["age"],
        categorical_columns=NORMALIZATION_COLUMNS,
    )

    print("\n=== Cleaned sample ===")
    print(
        service.df[["sex", "occupation", "income", "native_country", "age"]]
        .head(15)
        .to_string(index=False)
    )

    service.summary()

    print("\n=== Evaluation ===")
    eval_results = service.evaluate(GROUND_TRUTH)

    return {
        "delta": delta,
        "dataset": "UCI Adult",
        "sample_size": len(df),
        "cleaned_df_sample": service.df[
            ["sex", "occupation", "income", "native_country", "age"]
        ].head(15).to_dict(orient="records"),
        "detection_results": service.results["standardization"],
        "numeric_invalid": service.results["numeric_issues"],
        "llm_normalization": service.results.get("llm_normalization", {}),
        "validation_log": service.results["validation_log"],
        "evaluation": eval_results,
    }


all_outputs = []
sensitivity_results = []

for delta in DELTAS:
    output = run_evaluation(delta)
    all_outputs.append(output)

    overall = output["evaluation"]["overall"]
    sensitivity_results.append(
        {
            "delta": delta,
            "precision": overall["precision"],
            "recall": overall["recall"],
            "f1": overall["f1"],
            "fallback_rate": overall["fallback_rate"],
            "total_evaluated": overall["total_evaluated"],
        }
    )

sensitivity_df = pd.DataFrame(sensitivity_results)

print("\n=== THRESHOLD SENSITIVITY SUMMARY ===")
print(sensitivity_df.to_string(index=False))

with open("uci_adult_threshold_sensitivity.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "dataset": "UCI Adult",
            "deltas": DELTAS,
            "sensitivity": sensitivity_results,
            "runs": all_outputs,
        },
        f,
        indent=2,
        ensure_ascii=False,
        default=str,
    )

sensitivity_df.to_csv("uci_adult_threshold_sensitivity.csv", index=False)

print("\nSaved threshold sensitivity results:")
print("- uci_adult_threshold_sensitivity.json")
print("- uci_adult_threshold_sensitivity.csv")
