# ...existing code...
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from typing import List, Optional, Iterable


class EncodingService:
    """Encoding utilities for pandas DataFrames (label, one-hot, target)."""

    def __init__(self, df: Optional[pd.DataFrame] = None):
        self.df = df

    @staticmethod
    def detect_categorical_columns(df: pd.DataFrame) -> List[str]:
        return df.select_dtypes(include=['object', 'category']).columns.tolist()

    @staticmethod
    def label_encode(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
        out = df.copy()
        le = LabelEncoder()
        for col in columns:
            out[col] = le.fit_transform(out[col].astype(str))
        return out

    @staticmethod
    def one_hot_encode(df: pd.DataFrame, columns: Iterable[str], drop_first: bool = True) -> pd.DataFrame:
        return pd.get_dummies(df, columns=list(columns), drop_first=drop_first)

    @staticmethod
    def target_encode(df: pd.DataFrame, columns: Iterable[str], target: str) -> pd.DataFrame:
        te = TargetEncoder(cols=list(columns))
        out = df.copy()
        out[list(columns)] = te.fit_transform(out[list(columns)], out[target])
        return out

    def run(self, df: pd.DataFrame, methods: Optional[Iterable[str]] = None, target: Optional[str] = None) -> dict:
        """Algorithm-only runner. Returns dict of {method_name: transformed_df}."""
        if methods is None:
            methods = ["label", "onehot", "target"]

        results = {}
        cat_cols = self.detect_categorical_columns(df)
        if not cat_cols:
            results["original"] = df.copy()
            return results

        if "label" in methods:
            results["label"] = self.label_encode(df.copy(), cat_cols)

        if "onehot" in methods:
            results["onehot"] = self.one_hot_encode(df.copy(), cat_cols)

        if "target" in methods:
            if target is None:
                # fallback: use first categorical column as target if available
                target = cat_cols[0] if cat_cols else None
            if target is None or target not in df.columns:
                raise ValueError("Target column must be provided for target encoding and exist in dataframe.")
            results["target"] = self.target_encode(df.copy(), cat_cols, target)

        return results

    def run_from_file(self, input_path: str,
                      methods: Optional[Iterable[str]] = None,
                      target: Optional[str] = None,
                      output_dir: str = "Encoder/Output") -> dict:
        """Load CSV, run encoders, save outputs to output_dir and return results dict."""
        df = pd.read_csv(input_path)
        os.makedirs(output_dir, exist_ok=True)
        results = self.run(df, methods=methods, target=target)

        for name, out_df in results.items():
            out_path = os.path.join(output_dir, f"encoded_{name}.csv")
            out_df.to_csv(out_path, index=False)
        return results


if __name__ == "__main__":
    print("Encoder Demo Started")

    input_path = r"Input/encodertrain.csv"
    if not os.path.exists(input_path):
        raise SystemExit(f"Input file not found: {input_path}")

    tool = EncodingService()
    df = pd.read_csv(input_path)
    print("\n Original Data:")
    print(df.head())

    cat_cols = tool.detect_categorical_columns(df)
    print(f"\nDetected Categorical Columns: {cat_cols}")

    # run all methods; target falls back to first categorical col if not provided
    results = tool.run_from_file(input_path, methods=["label", "onehot", "target"], target=(cat_cols[0] if cat_cols else None))

    print("\nEncoded datasets saved to Encoder/Output:")
    for k in results:
        print(" -", k)
# ...existing code...