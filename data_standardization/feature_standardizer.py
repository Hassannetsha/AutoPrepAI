import pandas as pd
import json
import re
from groq import Groq

class FeatureStandardizer:
    def __init__(self, df: pd.DataFrame, client: Groq, model: str):
        """
        Generic data cleaner using LLM for categorical standardization and numeric validation.
        """
        self.df = df.copy()
        self.client = client
        self.model = model
        self.results = {"numeric_invalid": {}, "categorical": {}}

    def _parse_json(self, raw_text: str):
        """
        Robustly extract JSON from LLM output, even if extra text exists.
        """
        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON found in LLM output")
        return json.loads(match.group())

    def detect_categorical_issues(self, column: str, sample_size=100):
        values = self.df[column].dropna().astype(str).unique().tolist()[:sample_size]
        prompt = f"""
You are performing CATEGORICAL VALUE CLEANING.

Column: {column}

Your tasks:
1. Identify values that are INVALID (do not belong to this column).
2. Identify values that need STANDARDIZATION (same meaning, different surface form).

STRICT RULES:
- Do NOT merge distinct categories.
- Do NOT create higher-level abstractions.
- Do NOT group values unless they clearly refer to the SAME label.
- Do NOT include values that are already clean.

Return ONLY valid JSON with this schema:
[
  {{
    "type": "standardize" | "invalid",
    "canonical_value": "string | null",
    "variants": ["v1", "v2"],
    "reason": "short explanation"
  }}
]

Values:
{values}
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data quality expert. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        raw = completion.choices[0].message.content
        issues = self._parse_json(raw)
        self.results["categorical"][column] = issues
        return issues

    def detect_numeric_issues(self, column: str, sample_size=200):
        values = self.df[column].dropna().tolist()[:sample_size]
        prompt = f"""
You are analyzing a NUMERIC column for INVALID values only.

Column: {column}

Rules:
- ONLY flag values that are logically impossible or nonsensical.
- DO NOT flag outliers, rare values, or extreme but possible values.

Return ONLY valid JSON:
[
  {{
    "index": int,
    "value": number,
    "reason": "why this value is invalid"
  }}
]

Values (index, value):
{list(enumerate(values))}
"""
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a data quality expert. Return ONLY valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        raw = completion.choices[0].message.content
        issues = self._parse_json(raw)
        self.results["numeric_invalid"][column] = issues
        return issues

    def run_detection(self):
        """
        Run detection on all columns of the dataframe.
        """
        for col in self.df.columns:
            print(f"Analyzing column: {col}")
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.detect_numeric_issues(col)
            else:
                self.detect_categorical_issues(col)

    def apply_categorical_fixes(self, columns=None, apply_invalid=False):
        """
        Apply fixes suggested by the LLM for categorical columns.
        - columns: list of columns to fix, default all with issues
        - apply_invalid: if True, replace invalids with NaN
        """
        columns = columns or list(self.results["categorical"].keys())
        for col in columns:
            for issue in self.results["categorical"].get(col, []):
                if issue["type"] == "standardize":
                    for variant in issue["variants"]:
                        self.df[col] = self.df[col].replace(variant, issue["canonical_value"])
                elif apply_invalid and issue["type"] == "invalid":
                    for variant in issue["variants"]:
                        self.df[col] = self.df[col].replace(variant, pd.NA)

    def summary(self):
        """
        Print a clean summary of detected issues.
        """
        print("="*60)
        print("FINAL SUMMARY OF DETECTED ISSUES")
        print("="*60)
        print("\n### Numeric Invalid Values ###")
        print(json.dumps(self.results["numeric_invalid"], indent=2, ensure_ascii=False) or "None detected")
        print("\n### Categorical Standardization Issues ###")
        print(json.dumps(self.results["categorical"], indent=2, ensure_ascii=False) or "None detected")
        print("="*60)
        return self.results
