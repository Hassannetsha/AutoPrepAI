import pandas as pd
import json
import re
from groq import Groq
import unicodedata


class DataStandardizingService:
    def __init__(self, df: pd.DataFrame, client: Groq, model: str):
        """
        Generic data cleaner using LLM for categorical standardization and numeric validation.
        """
        self.df = df.copy()
        self.client = client
        self.model = model
        self.results = {"numeric_invalid": {}, "categorical": {}, "llm_normalization": {}}
        self._normalization_cache = {}

    def _parse_json_any(self, raw_text: str):
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        match_array = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if match_array:
            return json.loads(match_array.group())

        match_object = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match_object:
            return json.loads(match_object.group())

        raise ValueError("No valid JSON found in LLM output")

    def _parse_json(self, raw_text: str):
        """
        Robustly extract JSON from LLM output, even if extra text exists.
        """
        parsed = self._parse_json_any(raw_text)
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON array in LLM output")
        return parsed

    def _parse_json_object(self, raw_text: str):
        parsed = self._parse_json_any(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError("Expected JSON object in LLM output")
        return parsed

    def normalize_value(self, value):
        if not isinstance(value, str):
            return value

        original = value.strip()
        if not original:
            return original

        if original in self._normalization_cache:
            return self._normalization_cache[original]

        prompt = f"""
            You are normalizing ONE categorical value for a tabular dataset.

            Input value:
            {original}

            Rules:
            - Keep the SAME meaning.
            - Improve formatting only (case, spacing, obvious punctuation/encoding issues).
            - Do NOT translate.
            - Do NOT invent new information.

            Return ONLY valid JSON object:
            {{
            "normalized_value": "string",
            "reason": "short explanation"
            }}
            """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a data quality expert. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ]
            )
            raw = completion.choices[0].message.content
            parsed = self._parse_json_object(raw)
            normalized = parsed.get("normalized_value", original)
            if not isinstance(normalized, str) or not normalized.strip():
                normalized = original
            normalized = unicodedata.normalize("NFKC", normalized).strip()
        except Exception:
            normalized = original

        self._normalization_cache[original] = normalized
        return normalized

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

    def apply_llm_normalization(self, columns=None, max_unique_values=200):
        """
        Apply LLM-based value normalization for categorical columns.
        - columns: list of columns to normalize, default all non-numeric columns
        - max_unique_values: cap unique values per column to control API usage
        """
        if columns is None:
            columns = [
                col for col in self.df.columns
                if not pd.api.types.is_numeric_dtype(self.df[col])
            ]

        for col in columns:
            values = self.df[col].dropna().astype(str)
            unique_values = values.unique().tolist()[:max_unique_values]

            mapping = {}
            for raw_value in unique_values:
                normalized = self.normalize_value(raw_value)
                if normalized != raw_value:
                    mapping[raw_value] = normalized

            if mapping:
                self.df[col] = self.df[col].replace(mapping)

            self.results["llm_normalization"][col] = mapping

        return self.results["llm_normalization"]

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
