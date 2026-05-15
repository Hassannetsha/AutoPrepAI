import streamlit as st
import pandas as pd
from copy import deepcopy
from datetime import datetime

# IMPORT YOUR PIPELINE
from pipeline_builder import PipelineBuilder
from data_context import DataContext

# PAGE CONFIG
st.set_page_config(
    page_title="AutoPrepAI",
    layout="wide"
)

st.title("🤖 AutoPrepAI - Intelligent Data Preprocessing")

# SESSION STATE INIT
DEFAULT_STATE = {
    "df": None,
    "history": [],
    "mode": "💬 Chat Mode",
    "context": None
}

for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


# AVAILABLE OPERATIONS (friendly labels mapped to canonical intents)
AVAILABLE_OPERATIONS = {
    "Fix Data Types": "fix_data_types",
    "Resolve Inconsistencies": "remove_inconsistencies",
    "Correct Spelling": "correct_spelling",
    "Standardize Data": "standardize_data",
    "Remove Duplicates": "remove_duplicates",
    "Handle Missing Values": "handle_missing_values",
    "Detect/Remove Outliers": "remove_outliers",
    "Keep Outliers": "keep_outliers",
    "Select Features": "select_features",
    "Feature Selection (auto)": "feature_selection",
    "Scale Numerical Features": "scale_numerical",
    "Encode Categorical Features": "encode_categorical",
    "Suggest Features (LLM)": "suggest_features"
} 

# Default full intent set (used when user chooses to run all steps)
TARGET_INTENTS = [
    'handle_missing_values',
    'detect_outliers', 
    'remove_duplicates',
    'encode_categorical',
    'feature_selection',
    'fix_data_types',
    'remove_inconsistencies',
    'correct_spelling',
    'standardize_data',
    'scale_numerical',
    'feature_engineering'
]

AUTO_COMMAND = """
clean data automatically:
- handle missing values (with mean),
- remove duplicates,
- resolve inconsistencies,
- detect and remove outliers,
- correct spelling in categorical columns,
- standardize categorical values,
- engineer features for modeling
"""

Automade = [
    'handle_missing_values',
    'detect_outliers', 
    'remove_duplicates',
    'remove_inconsistencies',
    'correct_spelling',
    'standardize_data',
    'feature_engineering'
]

# SIDEBAR
st.sidebar.header("⚙️ Settings")

st.session_state.mode = st.sidebar.radio(
    "Choose Mode",
    [
        "🤖 Full Auto Mode",
        "💬 Chat Mode",
        "☑️ Manual Selection Mode"
    ]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.session_state.history.clear()
    st.sidebar.success("Dataset loaded successfully")

# WARNING: No data uploaded
if st.session_state.df is None:
    st.warning("⚠️ Please upload a CSV file to begin preprocessing")
    st.info("👈 Use the file uploader in the sidebar to get started")

# SHOW DATASET
if st.session_state.df is not None:
    st.subheader("📊 Current Dataset")
    st.dataframe(st.session_state.df, use_container_width=True)

# MANUAL MODE UI
selected_intents = []

if st.session_state.mode == "☑️ Manual Selection Mode":
    st.subheader("☑️ Select Preprocessing Steps")

    for label, intent in AVAILABLE_OPERATIONS.items():
        if st.checkbox(label):
            selected_intents.append(intent)

# CHAT INPUT
user_command = None

if st.session_state.mode == "💬 Chat Mode":
    user_command = st.chat_input("Describe what you want to do...", disabled=st.session_state.df is None)
    
    # Show warning if user tries to type without data
    if st.session_state.df is None:
        st.info("💡 Upload a dataset first to use chat mode")

elif st.session_state.mode == "🤖 Full Auto Mode":
    if st.button("🚀 Run Full Auto Cleaning", disabled=st.session_state.df is None):
        user_command = AUTO_COMMAND
    
    # Show warning if button is disabled
    if st.session_state.df is None:
        st.error("❌ Cannot run auto cleaning: No dataset uploaded")

elif st.session_state.mode == "☑️ Manual Selection Mode":
    if st.button("▶ Run Selected Steps", disabled=st.session_state.df is None):
        # If user didn't select any steps, default to running all target intents
        if not selected_intents:
            selected_intents = TARGET_INTENTS.copy()
            st.info("No steps selected — defaulting to run all target intents")
        user_command = "manual_selection"
    
    # Show warning if button is disabled
    if st.session_state.df is None:
        st.error("❌ Cannot run selected steps: No dataset uploaded")

# RUN PIPELINE
if user_command and st.session_state.df is not None:

    # -------- Context Creation --------
    context = DataContext(
        data=st.session_state.df.copy(),
        metadata={
            "has_text": True,
            "has_numeric": True,
            "has_categorical": True,
        }
    )

    # -------- Manual Mode: Inject Intents --------
    if st.session_state.mode == "☑️ Manual Selection Mode":
        context.metadata["nlp_done"] = True
        context.metadata["intents"] = [(intent,) for intent in selected_intents]
    elif st.session_state.mode == "🤖 Full Auto Mode" :
        context.metadata["nlp_done"] = True
        context.metadata["intents"] = [(intent,) for intent in Automade]

    pipeline = PipelineBuilder.build_default_pipeline()

    with st.spinner("Running preprocessing pipeline..."):
        final_context = pipeline.run(
            context,
            user_command="" if st.session_state.mode == "☑️ Manual Selection Mode" else user_command
        )

    # -------- Save History --------
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mode": st.session_state.mode,
        "command": user_command,
        "shape": final_context.data.shape,
        "logs": final_context.logs.copy(),
        "metadata": deepcopy(final_context.metadata),
        "data": final_context.data.copy()
    })

    # -------- Update State --------
    st.session_state.df = final_context.data
    st.session_state.context = final_context

    st.success("✅ Pipeline executed successfully")

# HISTORY VIEWER
if st.session_state.history:
    st.subheader("🕒 Operation History")

    for i, h in enumerate(reversed(st.session_state.history)):
        with st.expander(f"Step {len(st.session_state.history)-i} | {h['timestamp']}"):
            st.write("🔹 Mode:", h["mode"])
            st.write("📝 Command:", h["command"])
            st.write("📐 Shape:", h["shape"])

            st.markdown("**Logs:**")
            for log in h["logs"]:
                st.write("-", log)

            st.markdown("**Metadata:**")
            st.json(h["metadata"])

            st.markdown("**Data Preview:**")
            st.dataframe(h["data"].head())

# DOWNLOAD SECTION
if st.session_state.df is not None:
    st.subheader("💾 Download Cleaned Dataset")

    csv = st.session_state.df.to_csv(
        index=False,
        float_format="%.4f"  
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download CSV",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )
