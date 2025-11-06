import streamlit as st
import pandas as pd
import traceback

# Updated imports to match your refactored structure
from data_type_inconsistency_detector import DataTypeInconsistencyDetector
from data_type_inconsistency_resolver import DataResolver  # ← updated import

# ------------------------------------------------------------
# Streamlit App Configuration
# ------------------------------------------------------------
st.set_page_config(page_title="Data Inconsistency Detector & Resolver", layout="wide")

st.title("📊 Data Type Inconsistency Detector & Resolver")
st.write("Upload a CSV dataset, detect inconsistencies, and resolve them interactively.")

# ------------------------------------------------------------
# Initialize Session State
# ------------------------------------------------------------
for key in ["detector", "resolver", "df_original", "analysis_done"]:
    if key not in st.session_state:
        st.session_state[key] = None if key != "analysis_done" else False

# ------------------------------------------------------------
# Sidebar Configuration
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])


# ------------------------------------------------------------
# Handle File Upload
# ------------------------------------------------------------
if uploaded_file:
    try:
        # Reset session when a new file is uploaded
        if (
            st.session_state.df_original is None
            or uploaded_file.name != st.session_state.get("current_file_name")
        ):
            df = pd.read_csv(uploaded_file)
            st.session_state.df_original = df
            st.session_state.current_file_name = uploaded_file.name
            st.session_state.detector = None
            st.session_state.resolver = None
            st.session_state.analysis_done = False
            st.success("✅ File loaded successfully!")

        df = st.session_state.df_original

        # ------------------------------------------------------------
        # Dataset Overview
        # ------------------------------------------------------------
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        with st.expander("📋 Preview Dataset", expanded=False):
            st.dataframe(df.head(100), use_container_width=True)

        # ------------------------------------------------------------
        # Step 1: Inconsistency Detection
        # ------------------------------------------------------------
        st.header("🔍 Step 1: Detect Inconsistencies")

        if st.button("🚀 Run Inconsistency Detection", type="primary"):
            with st.spinner("Analyzing dataset..."):
                detector = DataTypeInconsistencyDetector()
                results = detector.analyze_dataframe(df)
                resolver = DataResolver(df, results)

                detector.results = results  # store results for access later
                st.session_state.detector = detector
                st.session_state.resolver = resolver
                st.session_state.analysis_done = True

            st.success("✅ Analysis complete!")
             

        if st.session_state.analysis_done and st.session_state.detector:
            detector = st.session_state.detector
            resolver = st.session_state.resolver

            st.subheader("📊 Detection Summary")

            # Show table of inconsistencies
            issues = [
                {"Column": col, "Recommended Type": res["recommended_type"]}
                for col, res in detector.results.items()
                if res["inconsistencies"]
            ]
            if issues:
                st.warning(f"⚠️ Found {len(issues)} columns with inconsistencies")
                st.dataframe(pd.DataFrame(issues))
            else:
                st.success("✅ No inconsistencies detected! Dataset is clean.")

            # ------------------------------------------------------------
            # Detailed Column Analysis
            # ------------------------------------------------------------
            st.subheader("🔎 Detailed Analysis")

            for col, result in detector.results.items():
                has_issues = bool(result["inconsistencies"])
                status_emoji = "⚠️" if has_issues else "✅"

                with st.expander(f"{status_emoji} Column: **{col}** ({result['recommended_type']})"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown(f"**Declared dtype:** `{result['declared_dtype']}`")
                        st.markdown(f"**Recommended type:** `{result['recommended_type']}`")
                        st.markdown(f"**Null count:** {result['null_count']}")

                    with c2:
                        if result["detected_types"]:
                            st.markdown("**Type Distribution:**")
                            for dtype, count in result["detected_types"].items():
                                pct = count / result["total_rows"] * 100 if result["total_rows"] else 0
                                st.progress(pct / 100, text=f"{dtype}: {count} ({pct:.1f}%)")

                    if has_issues:
                        inconsistent_values = result.get("inconsistent_values", [])
                        if inconsistent_values:
                            inconsistent_rows = df[df[col].isin(inconsistent_values)]

                            st.warning(f"⚠️ {len(inconsistent_rows)} rows contain inconsistent values in **{col}**")

                            # Display all inconsistent rows (or top 100)
                            st.dataframe(inconsistent_rows.head(100), use_container_width=True)

                            # Optional: allow user to download only those inconsistent rows
                            inconsistent_csv = inconsistent_rows.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label=f"📥 Download inconsistent rows for {col}",
                                data=inconsistent_csv,
                                file_name=f"inconsistent_{col}.csv",
                                mime="text/csv",
                                key=f"download_{col}"
                            )
                        else:
                            st.info("⚠️ Column has inconsistencies, but no specific inconsistent values were identified.")
                    else:
                        st.success("✓ No inconsistencies detected")

            # ------------------------------------------------------------
            # Step 2: Resolve Inconsistencies
            # ------------------------------------------------------------
            st.header("🔧 Step 2: Resolve Inconsistencies")

            inconsistent_cols = [col for col, r in detector.results.items() if r["inconsistencies"]]

            if inconsistent_cols:
                st.write("Choose how to resolve each problematic column:")

                for col in inconsistent_cols:
                    result = detector.results[col]
                    with st.expander(f"🧩 Resolve: **{col}** ({result['recommended_type']})"):
                        st.info(f"💡 Recommendation: convert to `{result['recommended_type']}`")

                        selected_strategy = st.selectbox(
                            f"Select resolution strategy for **{col}**:",
                            list(resolver.available_strategies().keys()),
                            key=f"strategy_{col}"
                        )

                        kwargs = {}
                        if "custom" in selected_strategy:
                            kwargs["value"] = st.text_input(f"Enter custom value for {col}:")

                        if st.button(f"✅ Apply {selected_strategy} to {col}", key=f"apply_{col}"):
                            new_df, msg = resolver.resolve(selected_strategy, col, **kwargs)
                            st.session_state.df_original = new_df
                            st.success(f"✅ Applied {selected_strategy} to {col}: {msg}")

                st.success("✅ All chosen resolutions applied!")

                st.header("📥 Step 3: Download Cleaned Data")
                cleaned_df = st.session_state.df_original
                csv_data = cleaned_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Cleaned Dataset (CSV)",
                    data=csv_data,
                    file_name="cleaned_dataset.csv",
                    mime="text/csv",
                    type="primary"
                )

            else:
                st.success("🎉 No inconsistencies to resolve! Dataset is clean.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.code(traceback.format_exc(), language="python")
else:
    st.info("👈 Upload a CSV file from the sidebar to get started!")
