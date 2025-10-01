import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🛠️ AutoPrepAI - Missing Values Handling Demo")

st.write("""
This app demonstrates different strategies to handle **missing values** in datasets:  
- **Mean Imputation**  
- **Median Imputation**  
- **Most Frequent Imputation**  
- **Constant (Fill with fixed value)**  
- **KNN Imputation**
""")

# -------------------------------
# Example dataset
# -------------------------------
example_data = {
    "Age": [25, 30, np.nan, 22, 28, np.nan],
    "Salary": [50000, 54000, 58000, np.nan, 60000, 62000],
    "Department": ["HR", np.nan, "IT", "Finance", np.nan, "IT"]
}
example_df = pd.DataFrame(example_data)

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
else:
    st.info("ℹ️ No file uploaded. Using built-in example dataset.")
    df = example_df

# -------------------------------
# Data Preview
# -------------------------------
st.subheader("📑 Data Preview")
st.dataframe(df.head())

st.write("**Missing values per column:**")
st.write(df.isna().sum())

# -------------------------------
# Choose imputation strategy
# -------------------------------
strategy = st.selectbox(
    "Choose an imputation strategy",
    ["Mean", "Median", "Most Frequent", "Constant", "KNN (3 Neighbors)"]
)

# -------------------------------
# Process Missing Values
# -------------------------------
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
non_numerical = [c for c in df.columns if c not in numerical_cols]
# cols = numerical_cols + non_numerical

df_imputed = None
imputer = SimpleImputer(strategy="most_frequent")
df_imputed = df.copy()
df_imputed[non_numerical] = imputer.fit_transform(df_imputed[non_numerical])
if strategy == "Mean":
    if numerical_cols:
        imputer = SimpleImputer(strategy="mean")
        
        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
    else:
        st.warning("⚠️ No numeric columns available for Mean imputation.")

elif strategy == "Median":
    if numerical_cols:
        imputer = SimpleImputer(strategy="median")

        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
    else:
        st.warning("⚠️ No numeric columns available for Median imputation.")

elif strategy == "Most Frequent":
    imputer = SimpleImputer(strategy="most_frequent")
    df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])

elif strategy == "Constant":
    fill_value = st.text_input("Enter constant value for missing data:", "missing")
    imputer = SimpleImputer(strategy="constant", fill_value=fill_value)
    df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])

elif strategy == "KNN (3 Neighbors)":
    if numerical_cols:
        imputer = KNNImputer(n_neighbors=3)

        df_imputed[numerical_cols] = imputer.fit_transform(df_imputed[numerical_cols])
    else:
        st.warning("⚠️ No numeric columns available for KNN imputation.")

# -------------------------------
# Show Results
# -------------------------------
if df_imputed is not None:
    st.subheader("✅ Imputed Data")
    st.dataframe(df_imputed.head(20))

    st.write("**Missing values after imputation:**")
    st.write(df_imputed.isna().sum())

    # Download option
    csv = df_imputed.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Imputed CSV", data=csv, file_name="imputed_dataset.csv", mime="text/csv")