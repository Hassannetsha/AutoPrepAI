# ...existing code...
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from typing import Any, Optional

class MissingValuesDemo:
    """Provides a Streamlit UI (runUI) and algorithm-only runner (run)."""

    def run(self, df: pd.DataFrame, strategy: str, fill_value: Optional[Any] = "missing", selected_cols: Optional[list[str]] = None) -> pd.DataFrame:
        """Run imputation algorithm on df according to strategy. Returns imputed DataFrame.
        This function does NOT use Streamlit and is safe to call from other code/tests.
        strategy: "Mean", "Median", "Most Frequent", "Constant", "KNN (3 Neighbors)"
        fill_value: used only for Constant strategy (applied to both numeric & non-numeric columns)
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        df_imputed = df.copy(deep=True)
        numerical_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
        non_numerical = [c for c in df_imputed.columns if c not in numerical_cols]

        # Handle non-numeric columns: default to most_frequent unless Constant requested
        if non_numerical:
            if strategy.strip().lower() == "constant":
                if(selected_cols):
                    cols_for_non_const = [c for c in selected_cols if c in non_numerical]
                else:
                    cols_for_non_const = non_numerical
                # only impute non-numeric columns that actually have missing values
                cols_for_non_const = [c for c in cols_for_non_const if df_imputed[c].isna().any()]
                if cols_for_non_const:
                    imputer_non = SimpleImputer(strategy="constant", fill_value=fill_value)
                    df_imputed[cols_for_non_const] = imputer_non.fit_transform(df_imputed[cols_for_non_const])
            elif strategy.strip().lower() == "most frequent":
                if(selected_cols):
                    cols_for_non_freq = [c for c in selected_cols if c in non_numerical]
                else:
                    cols_for_non_freq = non_numerical
                # only impute non-numeric columns that actually have missing values
                cols_for_non_freq = [c for c in cols_for_non_freq if df_imputed[c].isna().any()]
                if cols_for_non_freq:
                    imputer_non = SimpleImputer(strategy="most_frequent")
                    df_imputed[cols_for_non_freq] = imputer_non.fit_transform(df_imputed[cols_for_non_freq])
            
            
        # Handle numeric columns per strategy
        if strategy.strip().lower() == "mean":
            if selected_cols:
                cols_for_mean = [c for c in selected_cols if c in numerical_cols]
                if not cols_for_mean:   
                    cols_for_mean = numerical_cols
            else:
                cols_for_mean = numerical_cols
            # only impute numeric columns that actually have missing values
            cols_for_mean = [c for c in cols_for_mean if df_imputed[c].isna().any()]
            if cols_for_mean:
                imputer_num = SimpleImputer(strategy="mean")
                df_imputed[cols_for_mean] = imputer_num.fit_transform(df_imputed[cols_for_mean])


        elif strategy.strip().lower() == "median":
            if selected_cols:
                cols_for_median = [c for c in selected_cols if c in numerical_cols]
            else:
                cols_for_median = numerical_cols
            # only impute numeric columns that actually have missing values
            cols_for_median = [c for c in cols_for_median if df_imputed[c].isna().any()]
            if cols_for_median:
                imputer_num = SimpleImputer(strategy="median")
                df_imputed[cols_for_median] = imputer_num.fit_transform(df_imputed[cols_for_median])
        
        elif strategy.strip().lower() == "most frequent":
            if selected_cols:
                cols_for_most_freq = [c for c in selected_cols if c in numerical_cols]
            else:
                cols_for_most_freq = numerical_cols
            # only impute numeric columns that actually have missing values
            cols_for_most_freq = [c for c in cols_for_most_freq if df_imputed[c].isna().any()]
            if cols_for_most_freq:
                imputer_num = SimpleImputer(strategy="most_frequent")
                df_imputed[cols_for_most_freq] = imputer_num.fit_transform(df_imputed[cols_for_most_freq])


        elif strategy.strip().lower() == "constant":
            if selected_cols:
                cols_for_const = [c for c in selected_cols if c in numerical_cols]
            else:
                cols_for_const = numerical_cols
            # only impute numeric columns that actually have missing values
            cols_for_const = [c for c in cols_for_const if df_imputed[c].isna().any()]
            if cols_for_const:
                numeric_fill = pd.to_numeric(fill_value, errors="coerce")
                chosen_fill = numeric_fill if not np.isnan(numeric_fill) else fill_value
                imputer_num = SimpleImputer(strategy="constant", fill_value=chosen_fill)
                df_imputed[cols_for_const] = imputer_num.fit_transform(df_imputed[cols_for_const])

        elif strategy.strip().lower().startswith("knn"):
    # Apply KNN ONLY on selected numeric columns (or all numeric if none selected)
            if selected_cols:
                cols_for_knn = [c for c in selected_cols if c in numerical_cols]
            else:
                cols_for_knn = numerical_cols

            # only impute numeric columns that actually have missing values
            cols_for_knn = [c for c in cols_for_knn if df_imputed[c].isna().any()]

            if cols_for_knn:
                n_neighbors = 3
                try:
                    import re
                    m = re.search(r"(\d+)", strategy)
                    if m:
                        n_neighbors = int(m.group(1))
                except Exception:
                    pass

                imputer_num = KNNImputer(n_neighbors=n_neighbors)
                df_imputed[cols_for_knn] = imputer_num.fit_transform(df_imputed[cols_for_knn])

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return pd.DataFrame(df_imputed, columns=df.columns)

    def runUI(self):
        """Streamlit UI entrypoint that calls run(...) and displays results."""
        st.title("🛠️ AutoPrepAI - Missing Values Handling Demo")

        st.write("""
        This app demonstrates different strategies to handle **missing values** in datasets:  
        - **Mean Imputation**  
        - **Median Imputation**  
        - **Most Frequent Imputation**  
        - **Constant (Fill with fixed value)**  
        - **KNN Imputation**
        """)

        example_data = {
            "Age": [25, 30, np.nan, 22, 28, np.nan],
            "Salary": [50000, 54000, 58000, np.nan, 60000, 62000],
            "Department": ["HR", np.nan, "IT", "Finance", np.nan, "IT"]
        }
        example_df = pd.DataFrame(example_data)

        uploaded_file = st.file_uploader("📂 Upload your CSV file", type="csv")

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        else:
            st.info("ℹ️ No file uploaded. Using built-in example dataset.")
            df = example_df

        st.subheader("📑 Data Preview")
        st.dataframe(df.head())

        st.write("**Missing values per column:**")
        st.write(df.isna().sum())

        strategy = st.selectbox(
            "Choose an imputation strategy",
            ["Mean", "Median", "Most Frequent", "Constant", "KNN (3 Neighbors)"]
        )

        fill_value = "missing"
        if strategy == "Constant":
            fill_value = st.text_input("Enter constant value for missing data:", "missing")

        if st.button("Run Imputation"):
            try:
                imputed = self.run(df, strategy, fill_value=fill_value)
                st.subheader("✅ Imputed Data")
                st.dataframe(imputed.head(20))

                st.write("**Missing values after imputation:**")
                st.write(imputed.isna().sum())

                csv = imputed.to_csv(index=False).encode("utf-8")
                st.download_button("💾 Download Imputed CSV", data=csv, file_name="imputed_dataset.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Error during imputation: {e}")


if __name__ == "__main__":
    demo = MissingValuesDemo()
    # To run UI: `streamlit run missingvalues_demo.py` will execute runUI via Streamlit.
    # Here, call runUI if executed directly (streamlit will still import and call top-level)
    try:
        # detect if running under streamlit by checking for Streamlit runtime env
        if "streamlit" in __import__("sys").argv[0] or True:
            demo.runUI()
    except Exception:
        # fallback for direct programmatic run: show demonstration of run()
        df = pd.DataFrame({
            "A": [1, 2, np.nan, 4],
            "B": ["x", np.nan, "y", "z"]
        })
        print("Original:\n", df)
        out = demo.run(df, "Mean")
        print("Imputed:\n", out)
# ...existing code...