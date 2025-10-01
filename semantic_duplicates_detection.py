import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🔍 Query Deduplication using Semantic Similarity")

st.write("""
This app loads queries (from file or example dataset), computes embeddings using 
**Sentence Transformers**, and removes semantically similar queries based on cosine similarity.
""")

# -------------------------------
# Cache the model
# -------------------------------
@st.cache_resource
def load_model():
    # return SentenceTransformer("all-mpnet-base-v2")
    return SentenceTransformer("paraphrase-MiniLM-L6-v2")

# -------------------------------
# Example dataset
# -------------------------------
example_data = {
    "query": [
        "What is machine learning?",
        "Explain machine learning in simple terms",
        "How to cook pasta?",
        "Best way to boil pasta",
        "Weather in New York today",
        "New York weather forecast",
        "What is AI?",
        "Difference between AI and machine learning"
    ]
}
example_df = pd.DataFrame(example_data)

# -------------------------------
# File uploader
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload a CSV file with a 'query' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if "query" not in df.columns:
        st.error("CSV must contain a column named 'query'")
        st.stop()
    st.success(f"✅ Loaded dataset with {len(df)} rows.")
else:
    st.info("ℹ️ No file uploaded. Using built-in example dataset.")
    df = example_df

# -------------------------------
# Show preview
# -------------------------------
st.subheader("📑 Data Preview")
st.dataframe(df.head(20))

# -------------------------------
# Parameters
# -------------------------------
sample_size = st.number_input("Number of random samples to process:", min_value=5, max_value=len(df), value=min(8, len(df)), step=1)
threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.7, 0.05)

# -------------------------------
# Run button
# -------------------------------
if st.button("🚀 Run Deduplication"):
    # Take sample
    sample = df.sample(sample_size, random_state=42).reset_index(drop=True)

    # Load model (cached)
    with st.spinner("Loading model..."):
        model = load_model()

    # Embeddings
    with st.spinner("Generating embeddings..."):
        embeddings = model.encode(sample['query'].tolist(), show_progress_bar=True, convert_to_numpy=True, batch_size=64)

    # Cosine similarity
    with st.spinner("Computing similarity matrix..."):
        sim_matrix = cosine_similarity(embeddings)

    # Detect & display similar pairs
    st.subheader("🔗 Similar Query Pairs Found")
    pairs_found = False
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            if sim_matrix[i][j] > threshold:
                st.write(f"**Similarity {sim_matrix[i][j]:.2f}**")
                st.write(f"➡️ Query 1: {sample.iloc[i]['query']}")
                st.write(f"➡️ Query 2: {sample.iloc[j]['query']}")
                st.markdown("---")
                pairs_found = True
    if not pairs_found:
        st.info("No pairs found above the threshold.")

    # Deduplication
    to_drop = set()
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            if sim_matrix[i][j] > threshold:
                to_drop.add(j)

    deduped_sample = sample.drop(sample.index[list(to_drop)]).reset_index(drop=True)

    # Results
    st.subheader("✅ Deduplication Results")
    st.write(f"Original size: **{len(sample)}**")
    st.write(f"After deduplication: **{len(deduped_sample)}**")

    st.dataframe(deduped_sample)

    # Download option
    csv = deduped_sample.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Download Deduplicated CSV", data=csv, file_name="deduped_queries.csv", mime="text/csv")
