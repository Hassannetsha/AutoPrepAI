"""
Streamlit UI components and handlers.
"""

import streamlit as st
import pandas as pd
from typing import List, Dict, Tuple, Optional

class StreamlitUI:
    """Handles all Streamlit UI components and interactions."""
    
    def __init__(self, intent_processor):
        """
        Initialize Streamlit UI.
        
        Args:
            intent_processor: Instance of IntentProcessor
        """
        self.intent_processor = intent_processor
        self.setup_page()
        
    def setup_page(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="🧠 AutoPrepAI — ML Intent Understanding",
            layout="centered"
        )
        st.title("💬 AutoPrepAI — ML Transformer NLP (No Rules)")
        
    def render_header(self):
        """Render the application header and description."""
        st.markdown("""
        ### 🧩 Type what you want AutoPrepAI to do (
        Examples:
        - "Handle missing values using median and remove duplicates by column ID"
        - "Normalize salary and age columns, then detect outliers and drop extreme rows"
        """)
        
    def handle_file_upload(self) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
        """
        Handle dataset file upload and preview.
        
        Returns:
            Tuple of (dataframe, column_names) or (None, None)
        """
        st.markdown("### 📂 Upload Your Dataset (optional)")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is None:
            st.info("💡 Upload a dataset to allow column-aware ML extraction (optional).")
            return None, None
            
        try:
            # Read the file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            df_columns = df.columns.tolist()
            
            # Show success message and preview
            st.success(
                f"✅ File uploaded: {uploaded_file.name} — "
                f"{len(df)} rows, {len(df_columns)} columns."
            )
            
            with st.expander("📊 Preview Dataset"):
                st.dataframe(df.head(8))
                st.write(f"**Columns:** {', '.join(df_columns)}")
                
            return df, df_columns
            
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            return None, None
            
    def display_intent_result(self, step_number: int, result: Dict):
        """
        Display the results for a single detected intent.
        
        Args:
            step_number: The step number in the sequence
            result: Dict containing the intent detection results
        """
        st.markdown(f"### Step {step_number}")
        st.markdown(f"**➡️ Clause:** `{result['clause']}`")
        st.write(
            f"**Predicted Intent:** `{result['matched_intent']}` — "
            f"Confidence: `{result['intent_score']:.2f}`"
        )
        
        if result["params"]:
            st.write("🧩 Extracted Parameters:")
            for key, value in result["params"].items():
                st.markdown(f"- **{key}** → `{value}`")
        else:
            st.write("No parameters detected.")
        
        st.markdown("---")
        
    def handle_user_input(self, df: Optional[pd.DataFrame], df_columns: Optional[List[str]]):
        """
        Handle user input processing and result display.
        
        Args:
            df: Optional pandas DataFrame if file was uploaded
            df_columns: Optional list of column names from uploaded file
        """
        st.markdown("---")
        user_input = st.text_area(
            "✍️ Enter your command:",
            height=150,
            placeholder="e.g., handle missing values using median and remove duplicates by column ID"
        )

        if not st.button("🔍 Understand Intents"):
            st.info("💡 Enter a command and click the button to analyze it.")
            return

        if not user_input.strip():
            st.warning("⚠️ Please enter a command first.")
            return

        # Process the input
        results, cleaned = self.intent_processor.process_intents(user_input, df_columns)
        
        # Display preprocessing results
        st.subheader("🧹 Preprocessing & Splitting")
        st.write(f"**Cleaned (lemmatized):** `{cleaned}`")
        
        # Display detected intents
        st.subheader("🎯 Detected Intents and Parameters (ML-driven)")
        for i, result in enumerate(results, 1):
            self.display_intent_result(i, result)
        
        st.success("✅ Intents detected.")
        
        # Show preview if dataset is available
        if df is not None:
            st.markdown("### 🎬 What will happen to your data:")
            st.info("This is a preview of the detected operations. Actual execution coming soon!")
            
    def run(self):
        """Run the Streamlit application."""
        self.render_header()
        df, df_columns = self.handle_file_upload()
        self.handle_user_input(df, df_columns)