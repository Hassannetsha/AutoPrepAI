# ...existing code...
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Any, Optional, List

class Scaler:
    """Wrapper class providing scaling utilities for pandas DataFrames."""

    def standard(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply StandardScaler to numeric columns and return a new DataFrame.

        If `columns` is provided, only those numeric columns (intersection) will be scaled.
        """
        df_out = df.copy()
        numeric_cols = list(df_out.select_dtypes(include=['number']).columns)
        if columns:
            cols = [c for c in columns if c in numeric_cols]
        else:
            cols = numeric_cols
        if not cols:
            print("📏 No numeric columns selected for Standard scaler.")
            return df_out
        scaler = StandardScaler()
        df_out.loc[:, cols] = scaler.fit_transform(df_out[cols])
        print(f"📏 Applied Standard Scaler on: {cols}")
        return df_out

    def minmax(self, df: pd.DataFrame, columns: Optional[List[str]] = None, feature_range: Tuple[float, float] = (0, 1)) -> pd.DataFrame:
        """Apply MinMaxScaler with given feature_range to numeric columns.

        If `columns` provided, only those numeric columns (intersection) will be scaled.
        """
        df_out = df.copy()
        numeric_cols = list(df_out.select_dtypes(include=['number']).columns)
        if columns:
            cols = [c for c in columns if c in numeric_cols]
        else:
            cols = numeric_cols
        if not cols:
            print("📊 No numeric columns selected for MinMax scaler.")
            return df_out
        scaler = MinMaxScaler(feature_range=feature_range)
        df_out.loc[:, cols] = scaler.fit_transform(df_out[cols])
        print(f"📊 Applied MinMax Scaler with range {feature_range} on: {cols}")
        return df_out

    def robust(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Apply RobustScaler to numeric columns.

        If `columns` provided, only those numeric columns (intersection) will be scaled.
        """
        df_out = df.copy()
        numeric_cols = list(df_out.select_dtypes(include=['number']).columns)
        if columns:
            cols = [c for c in columns if c in numeric_cols]
        else:
            cols = numeric_cols
        if not cols:
            print("🧱 No numeric columns selected for Robust scaler.")
            return df_out
        scaler = RobustScaler()
        df_out.loc[:, cols] = scaler.fit_transform(df_out[cols])
        print(f"🧱 Applied Robust Scaler on: {cols}")
        return df_out

    def scale(self, df: pd.DataFrame, method: str = "standard", columns: Optional[List[str]] = None, **kwargs: Any) -> pd.DataFrame:
        """Dispatch helper: method in {'standard','minmax','robust'}.

        Pass `columns` to restrict scaling to specific columns if provided.
        """
        method = method.lower()
        if method == "standard":
            return self.standard(df, columns=columns)
        if method == "minmax":
            return self.minmax(df, columns=columns, feature_range=kwargs.get("feature_range", (0, 1)))
        if method == "robust":
            return self.robust(df, columns=columns)
        raise ValueError(f"Unknown scaling method: {method}")


if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data.csv")
    print("✅ Dataset loaded successfully!")
    print(df.head())

    scaler = Scaler()

    # 1️⃣ Standard Scaler
    df_standard = scaler.scale(df.copy(), method="standard")

    # 2️⃣ MinMax Scaler
    df_minmax = scaler.scale(df.copy(), method="minmax", feature_range=(0, 1))

    # 3️⃣ Robust Scaler
    df_robust = scaler.scale(df.copy(), method="robust")

    # Print sample results
    print("\n🔹Standard Scaled Data:")
    print(df_standard.head())

    print("\n🔹MinMax Scaled Data:")
    print(df_minmax.head())

    print("\n🔹Robust Scaled Data:")
    print(df_robust.head())
# ...existing code...