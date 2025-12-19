import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ----------- Standard Scaler -----------
def standard_scaler(df, selected_cols=None):
    scaler = StandardScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if selected_cols:
        cols_to_scale = [c for c in selected_cols if c in numeric_cols]
    else:
        cols_to_scale = numeric_cols

    if cols_to_scale:
        df.loc[:, cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df



# ----------- MinMax Scaler -----------
def minmax_scaler(df, selected_cols=None, feature_range=(0, 1)):
    scaler = MinMaxScaler(feature_range=feature_range)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if selected_cols:
        cols_to_scale = [c for c in selected_cols if c in numeric_cols]
    else:
        cols_to_scale = numeric_cols

    if cols_to_scale:
        df.loc[:, cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df


# ----------- Robust Scaler -----------
def robust_scaler(df, selected_cols=None):
    scaler = RobustScaler()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    if selected_cols:
        cols_to_scale = [c for c in selected_cols if c in numeric_cols]
    else:
        cols_to_scale = numeric_cols

    if cols_to_scale:
        df.loc[:, cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    return df



# ----------- Example Usage -----------
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv("data.csv")
    print(df.head())

    # Standard Scaler
    df_standard = standard_scaler(df.copy(), selected_cols=['Age'])

    # MinMax Scaler
    df_minmax = minmax_scaler(df.copy(), selected_cols=['Salary'], feature_range=(0, 1))

    # Robust Scaler
    df_robust = robust_scaler(df.copy(), selected_cols=['Age', 'Salary'])

    print("\n🔹Standard Scaled Data:")
    print(df_standard.head())

    print("\n🔹MinMax Scaled Data:")
    print(df_minmax.head())

    print("\n🔹Robust Scaled Data:")
    print(df_robust.head())
