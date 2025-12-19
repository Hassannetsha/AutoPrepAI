import dspy
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import re

class SuggestFeatures(dspy.Signature):
    """Suggest meaningful new features for a dataset.
    Outputs a set of suggested new column definitions, one per line in the format:
    name: description | code: pandas expression
    """
    dataset_columns = dspy.InputField(desc="Available column names (comma-separated)", default="")
    sample_rows = dspy.InputField(desc="Sample rows as JSON (first N rows)", default="")
    top_n = dspy.InputField(desc="Number of suggestions to return", default="5")
    suggested_features = dspy.OutputField(desc="Suggested features, one per line (name: description | code: ...)")

class FeatureEngineer:
    """Feature engineering utilities encapsulated as a class to use from the pipeline.

    Example:
        fe = FeatureEngineer()
        df_new = fe.engineer(df, suggested_features_str)
    """

    @staticmethod
    def fix_column_references(code: str, df_columns: list) -> str:
        """Fix bare column names in code to use df['column'] syntax.
        """
        fixed_code = code
        sorted_columns = sorted(df_columns, key=len, reverse=True)
        for col in sorted_columns:
            pattern = r'(?<!df\[\')(?<!df\[")(?<!\w)' + re.escape(col) + r'(?!\w)(?!\'\])(?!\"\])'
            replacement = f"df['{col}']"
            fixed_code = re.sub(pattern, replacement, fixed_code)
        return fixed_code

    def apply(self, DataFrame: pd.DataFrame, suggested_features: str) -> tuple[pd.DataFrame, int]:
        """Apply feature engineering described in `suggested_features` to the DataFrame.

        `suggested_features` is expected to be one feature per line in the
        format: `name: description | code: pandas_expression`.

        Returns a tuple (new_dataframe, features_added_count).
        """
        df = DataFrame.copy()
        df_columns = df.columns.tolist()
        features_added = 0

        lines = suggested_features.strip().split("\n")
        for line in lines:
            if not line.strip():
                continue
            try:
                if "| code:" not in line:
                    print(f"Skipping invalid format: {line}")
                    continue
                name_desc, code_part = line.split("| code:", 1)
                if ":" not in name_desc:
                    print(f"Skipping line without name: {line}")
                    continue
                name, _ = name_desc.split(":", 1)
                name = name.strip()
                name = re.sub(r'^\d+\.\s*', '', name)
                code = code_part.strip()

                print(f"\n{'='*50}")
                print(f"Processing feature: {name}")
                print(f"Original code: {code}")

                fixed_code = self.fix_column_references(code, df_columns)
                print(f"Fixed code: {fixed_code}")

                eval_context = {"df": df, "pd": pd, "np": np, "numpy": np}
                df[name] = eval(fixed_code, eval_context)
                features_added += 1
                print(f"✓ Successfully added feature: {name}")
            except Exception as e:
                print(f"✗ Error processing line '{line}'")
                print(f"  Error: {e}")
                import traceback
                print(f"  Traceback: {traceback.format_exc()}")
                continue

        print(f"\n{'='*50}")
        print(f"Total features added: {features_added}")
        return df, features_added

    def engineer(self, df: Any, suggested_features: str) -> tuple[pd.DataFrame, int]:
        if not isinstance(df, pd.DataFrame):
            raise ValueError("First argument must be a pandas DataFrame")
        if not suggested_features or not suggested_features.strip():
            print("No features to apply")
            return df, 0
        print(f"{'='*50}")
        print(f"Starting feature engineering")
        print(f"Dataframe shape: {df.shape}")
        print(f"Dataframe columns: {list(df.columns)}")
        print(f"{'='*50}")
        print(f"Features to apply:")
        print(suggested_features)
        print(f"{'='*50}")
        return self.apply(df, suggested_features)


# Backwards-compatible wrappers
def apply_feature_engineering_agent(DataFrame, suggested_features: str) -> tuple[pd.DataFrame, int]:
    fe = FeatureEngineer()
    return fe.apply(DataFrame, suggested_features)


def engineer_features(df: Any, suggested_features: str) -> tuple[pd.DataFrame, int]:
    fe = FeatureEngineer()
    return fe.engineer(df, suggested_features)