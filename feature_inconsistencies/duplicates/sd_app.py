import streamlit as st
import pandas as pd
import io
from sentence_transformers import SentenceTransformer
from semantic_duplicate_remover import SemanticDuplicateRemover  # <- import your class

# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------
st.title("AI-Powered Semantic Duplicate Cleaner")
st.write("Upload a CSV file, select a text column, and automatically remove semantic duplicate rows.")

# File upload
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of your dataset:")
    st.dataframe(df.head())

    # Select column
    text_column = st.selectbox("Select the text column to check for duplicates", df.columns)

    # Threshold slider
    threshold = st.slider("Similarity threshold", 0.5, 0.95, 0.8)

    # Model cache
    @st.cache_resource
    def load_remover(threshold_value):
        return SemanticDuplicateRemover(
            model_name="paraphrase-MiniLM-L6-v2",
            threshold=threshold_value,
            k_neighbors=10
        )

    remover = load_remover(threshold)

    if st.button("Find and Remove Duplicates"):
        with st.spinner("Detecting semantic duplicates..."):
            cleaned_df, duplicates_df = remover.remove_duplicates(df, text_column)

        # Show results
        if not duplicates_df.empty:
            st.subheader("Detected Semantic Duplicates")
            st.dataframe(duplicates_df[["text_1", "text_2", "similarity"]].head(20))
            st.write(f"Total duplicates found: {len(duplicates_df)}")
        else:
            st.success("✅ No semantic duplicates found above the threshold.")

        # Show cleaned dataset
        st.subheader("Cleaned Dataset (duplicates removed)")
        st.dataframe(cleaned_df.head())

        # Download button
        csv_buffer = io.StringIO()
        cleaned_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Cleaned CSV",
            data=csv_buffer.getvalue(),
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )
