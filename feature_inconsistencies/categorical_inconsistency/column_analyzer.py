import pandas as pd
from .utils import looks_non_categorical, normalize_text
from .constants import CATEGORICAL_UNIQUE_RATIO_THRESHOLD, INVALID_CATEGORY_VALUES

class ColumnAnalyzer:
    def __init__(self, category_detector, semantic_matcher):
        self.category_detector = category_detector
        self.semantic_matcher = semantic_matcher

    def analyze_column(self, df: pd.DataFrame, column: str):
        series = df[column].astype(str).dropna()
        normalized = normalize_text(series)
        # print(column)

        invalids = []
        issues = []
        inconsistent_values = []
        inconsistent_indices = []

        # Skip IDs / pattern-based
        if looks_non_categorical(normalized):
            return {
                "column": column,
                "status": "skipped",
                "reason": "Pattern-based (likely ID or code).",
                "issues": issues,
                "inconsistent_values": inconsistent_values,
                "inconsistent_indices": inconsistent_indices,
                "message": "Skipped column. Invalid values detected." if invalids else "Skipped column."
            }

        # Check invalid values
        invalids = [v for v in series.str.lower().unique() if v in INVALID_CATEGORY_VALUES]
        if invalids:
            issues.append({"type": "invalid_values", "values": invalids})
            for val in invalids:
                idxs = df[df[column].astype(str).str.lower() == val].index.tolist()
                inconsistent_values.extend([val] * len(idxs))
                inconsistent_indices.extend(idxs)

        # Skip numeric or too unique columns
        unique_ratio = normalized.nunique() / len(normalized)
        if unique_ratio > CATEGORICAL_UNIQUE_RATIO_THRESHOLD:
            return {
                "column": column,
                "status": "skipped",
                "reason": "Too unique to be categorical.",
                "issues": issues,
                "inconsistent_values": inconsistent_values,
                "inconsistent_indices": inconsistent_indices,
                "message": "Skipped column. Invalid values detected." if invalids else "Skipped column."
            }

        # --- Use only Category Detector (fuzzy) ---
        # cat_result = self.category_detector.detect(df, column)
        # issues.extend(cat_result.get("issues", []))
        # inconsistent_values.extend(cat_result.get("inconsistent_values", []))
        # inconsistent_indices.extend(cat_result.get("inconsistent_indices", []))

        # --- Semantic similarity part commented out ---
        unique_vals = normalized.unique().tolist()
        similar_groups = self.semantic_matcher.find_similar_groups(unique_vals)
        if similar_groups:
            issues.append({"type": "semantic_similarity", "groups": similar_groups})
            for group in similar_groups:
                for val in group:
                    idxs = df[df[column].astype(str).str.lower() == val].index.tolist()
                    inconsistent_values.extend([val] * len(idxs))
                    inconsistent_indices.extend(idxs)

        return {
            "column": column,
            "status": "checked",
            "issues": issues,
            "inconsistent_values": inconsistent_values,
            "inconsistent_indices": inconsistent_indices,
            "message": "Inconsistencies detected." if issues else "No issues found."
        }
