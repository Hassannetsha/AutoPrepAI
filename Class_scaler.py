# ...existing code...
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from typing import Tuple, Any, Optional

class Scaler:
    """Wrapper class providing scaling utilities for pandas DataFrames."""

    def standard(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply StandardScaler to numeric columns and return a new DataFrame."""
        df_out = df.copy()
        scaler = StandardScaler()
        numeric_cols = df_out.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df_out.loc[:, numeric_cols] = scaler.fit_transform(df_out[numeric_cols])
        print(f"📏 Applied Standard Scaler on: {list(numeric_cols)}")
        return df_out

    def minmax(self, df: pd.DataFrame, feature_range: Tuple[float, float] = (0, 1)) -> pd.DataFrame:
        """Apply MinMaxScaler with given feature_range to numeric columns."""
        df_out = df.copy()
        scaler = MinMaxScaler(feature_range=feature_range)
        numeric_cols = df_out.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df_out.loc[:, numeric_cols] = scaler.fit_transform(df_out[numeric_cols])
        print(f"📊 Applied MinMax Scaler with range {feature_range} on: {list(numeric_cols)}")
        return df_out

    def robust(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply RobustScaler to numeric columns."""
        df_out = df.copy()
        scaler = RobustScaler()
        numeric_cols = df_out.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            df_out.loc[:, numeric_cols] = scaler.fit_transform(df_out[numeric_cols])
        print(f"🧱 Applied Robust Scaler on: {list(numeric_cols)}")
        return df_out

    def scale(self, df: pd.DataFrame, method: str = "standard", **kwargs: Any) -> pd.DataFrame:
        """Dispatch helper: method in {'standard','minmax','robust'}."""
        method = method.lower()
        if method == "standard":
            return self.standard(df)
        if method == "minmax":
            return self.minmax(df, feature_range=kwargs.get("feature_range", (0, 1)))
        if method == "robust":
            return self.robust(df)
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