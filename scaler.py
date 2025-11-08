import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ----------- Standard Scaler -----------
def standard_scaler(df):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df.loc[:, numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"📏 Applied Standard Scaler on: {list(numeric_cols)}")
    return df


# ----------- MinMax Scaler -----------
def minmax_scaler(df, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    numeric_cols = df.select_dtypes(include=['number']).columns
    df.loc[:, numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"📊 Applied MinMax Scaler with range {feature_range} on: {list(numeric_cols)}")
    return df


# ----------- Robust Scaler -----------
def robust_scaler(df):
    scaler = RobustScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns
    df.loc[:, numeric_cols] = scaler.fit_transform(df[numeric_cols])
    print(f"🧱 Applied Robust Scaler on: {list(numeric_cols)}")
    return df


# ----------- Example Usage -----------
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data.csv")
    print("✅ Dataset loaded successfully!")
    print(df.head())

    # 1️⃣ Standard Scaler
    df_standard = standard_scaler(df.copy())

    # 2️⃣ MinMax Scaler
    df_minmax = minmax_scaler(df.copy())

    # 3️⃣ Robust Scaler
    df_robust = robust_scaler(df.copy())

    # Print sample results
    print("\n🔹Standard Scaled Data:")
    print(df_standard.head())

    print("\n🔹MinMax Scaled Data:")
    print(df_minmax.head())

    print("\n🔹Robust Scaled Data:")
    print(df_robust.head())
