# resolver/category_resolver.py
import pandas as pd
import numpy as np
from .utils import normalize_text

class CategoryInconsistencyResolver:
    def __init__(self):
        pass

    def resolve(self, df: pd.DataFrame, column: str, report: dict, skip_groups: list = None):
        if report["status"] == "skipped":
            return df, f"Skipped '{column}' — not categorical."

        df[column] = normalize_text(df[column])
        skip_groups = skip_groups or []

        if report["issues"]:
            for issue in report["issues"]:
                if issue["type"] == "invalid_values":
                    df[column] = df[column].replace(issue["values"], np.nan)

                elif issue["type"] in ["similar_categories", "semantic_similarity"]:
                    for group in issue.get("groups", []):
                        if group in skip_groups:
                            continue
                        replacement = self._choose_replacement(df[column], group)
                        df[column] = df[column].apply(
                            lambda x: replacement if pd.notna(x) and x in group else x
                        )
            return df, f"Resolved inconsistencies in '{column}'."
        return df, "No issues were found"


    def _choose_replacement(self, series, group):
        freq = {v: (series == v).sum() for v in group}
        max_freq = max(freq.values())
        candidates = [k for k, v in freq.items() if v == max_freq]
        return min(candidates, key=len)
