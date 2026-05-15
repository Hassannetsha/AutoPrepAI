import os
import re
import time
from typing import List, Optional, Dict, Any

import streamlit as st
import pandas as pd
import dspy
from services.feature_engineering_service import *
import json
from api_key_manager import get_key_manager, get_api_key, rotate_api_key

# Initialize the key manager
_key_manager = get_key_manager()
os.environ['GROQ_API_KEY'] = _key_manager.get_current_key()

class NLPService:
    """AutoPrepAI application class: encapsulates DSPy setup, pipeline creation, data loading and Streamlit UI.

    Design notes to respect SOLID:
    - Single Responsibility: each method has one clear responsibility (setup, load data, build pipeline, render UI).
    - Open/Closed: pipeline construction can accept training data and demos without modifying the class.
    - Liskov Substitution: public API is stable (run()).
    - Interface Segregation: consumer (new.py) uses only run(); internals are split into helpers.
    - Dependency Inversion: external deps (env, streamlit, dspy) are injected via environment or parameters where practical.
    """
    
    _lm = None
    _pipeline = None
    _training_data_loaded = False

    def __init__(self, groq_api_key=None, training_csv="intents_augmented.csv"):
        self.training_csv = training_csv
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        # Initialize shared resources once
        if NLPService._lm is None:
            self._init_lm()
        self.lm = NLPService._lm
        if NLPService._pipeline is None:
            self._init_pipeline()

    def _init_lm(self):
        api_key = _key_manager.get_current_key()
        os.environ["GROQ_API_KEY"] = api_key
        lm = dspy.LM(
            model="groq/llama-3.3-70b-versatile",
            api_key=api_key,
            max_tokens=1000
        )
        dspy.settings.configure(lm=lm)
        NLPService._lm = lm
        self.lm = lm
        print(f"✅ Using API Key #{_key_manager.current_index + 1}/{_key_manager.get_total_keys_count()}")

    def _init_pipeline(self):
        try:
            training_data = pd.read_csv(self.training_csv)
        except Exception:
            training_data = None
        NLPService._pipeline = self.build_pipeline(training_data)

    def build_pipeline(self, training_data):
        """No longer uses @st.cache_resource - plain method."""
        if training_data is not None and len(training_data) > 0:
            return NLPService.OptimizedIntentPipeline(training_examples=training_data)
        return NLPService.OptimizedIntentPipeline(training_examples=None)


    def setup_dspy(_self) -> dspy.LM:
        """Initialize and cache DSPy LM resource with key rotation support."""
        api_key = None
        max_retries = _key_manager.get_total_keys_count()
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Get current API key
                api_key = _key_manager.get_current_key()
                if not api_key:
                    st.warning("⚠️ No API keys available!")
                    st.stop()
                
                # Configure LM with current key
                lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
                dspy.settings.configure(lm=lm)
                _self.lm = lm
                
                # Show key status
                st.sidebar.info(
                    f"✅ Using API Key #{_key_manager.current_index + 1}/{_key_manager.get_total_keys_count()}\n"
                    f"Available keys: {_key_manager.get_available_keys_count()}"
                )
                
                return lm
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's a rate limit error
                if "rate" in error_msg or "quota" in error_msg or "limit" in error_msg or "429" in error_msg:
                    st.warning(f"⚠️ API Key #{_key_manager.current_index + 1} rate limited. Rotating...")
                    _key_manager.mark_key_failed()
                    
                    # Try next key
                    try:
                        api_key = _key_manager.rotate_key()
                        os.environ["GROQ_API_KEY"] = api_key
                        retry_count += 1
                        time.sleep(1)  # Brief delay before retry
                        continue
                    except RuntimeError as rotate_err:
                        st.error(f"❌ {rotate_err}")
                        st.stop()
                else:
                    # Other error
                    st.error(f"❌ Setup Error: {e}")
                    st.stop()

    def load_training_data(_self) -> Optional[pd.DataFrame]:
        """Load training CSV if present; cache result."""
        try:
            df = pd.read_csv(_self.training_csv)
            return df
        except FileNotFoundError:
            st.sidebar.warning(f"⚠️ Training file '{_self.training_csv}' not found. Using basic mode.")
            return None
        
    # ---------- DSPy Signatures ----------
    # Keep these as inner classes for encapsulation but accessible by dspy
    class SplitIntoTasks(dspy.Signature):
        """Split a complex user command into distinct preprocessing steps.
        Rules:
        - If the user mentions multiple operations (e.g., 'handle missing values and remove outliers'),
            split into one line per operation.
        - If the user mentions columns (e.g., 'for age and salary'), keep them in the same line.
        - If no columns are mentioned, keep it as a single task that applies to all columns.
        """
        user_command = dspy.InputField(desc="The user command (one or more preprocessing actions)")
        dataset_columns = dspy.InputField(desc="Available dataset columns (comma-separated)", default="")
        tasks = dspy.OutputField(desc="Each preprocessing action on a separate line.")

    class ClassifyIntent(dspy.Signature):
        """Classify the data preprocessing intent from a task description.
    
        Available intents:
        - handle_missing_values: Fill, impute, or handle missing/null/NaN values
        - detect_outliers / remove_outliers: Identify and remove outliers, anomalies, or extreme values
        - keep_outliers: Preserve or keep outliers in the data
        - remove_duplicates: Remove duplicate rows or entries
        - encode_categorical: Convert categorical/text columns to numeric
        - feature_selection / select_features: Select important features or columns for modeling
        - fix_data_types / remove_inconsistencies: Detect and resolve inconsistent types (dates, numbers, booleans)
        - correct_spelling: Fix spelling errors in categorical/text columns
        - standardize_data: Normalize or standardize categorical values
        - scale_numerical: Scale numerical columns (standard, minmax, robust)
        - feature_engineering / suggest_features: Suggest and/or apply new derived features
        """
        task = dspy.InputField(desc="A single preprocessing task description")
        intent = dspy.OutputField(desc="The intent category (must be one of: handle_missing_values, detect_outliers, remove_outliers, keep_outliers, remove_duplicates, encode_categorical, feature_selection, select_features, fix_data_types, remove_inconsistencies, correct_spelling, standardize_data, scale_numerical, feature_engineering, suggest_features)")
        confidence = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
        reasoning = dspy.OutputField(desc="Brief explanation for the classification")

    class ExtractParameters(dspy.Signature):
        """Extract specific parameters from a preprocessing task.
    
        Look for:
        - If the user says "for column X" or "for X and Y", list only those columns.
        - If no specific columns are mentioned, set columns='none' (it will mean all columns later).
        - Methods like: mean, median, mode, IQR, z-score, one-hot, label encoding
        - Numeric values or thresholds
        """
        task = dspy.InputField(desc="A preprocessing task description")
        dataset_columns = dspy.InputField(desc="Available column names (comma-separated)", default="")
        intent = dspy.InputField(desc="The classified intent")
        columns = dspy.OutputField(desc="Column names mentioned (comma-separated), or 'none' if not specified")
        method = dspy.OutputField(desc="Method/algorithm mentioned (e.g., mean, median, IQR), or 'none'")
        other_params = dspy.OutputField(desc="Other parameters as key:value pairs (comma-separated), or 'none'")
    
    class OptimizedIntentPipeline(dspy.Module):
        """Intent understanding pipeline with few-shot learning from training data"""

        def __init__(self, training_examples=None):
            super().__init__()
            self.split_tasks = dspy.ChainOfThought(NLPService.SplitIntoTasks)
            self.classify = dspy.ChainOfThought(NLPService.ClassifyIntent)
            self.extract_params = dspy.ChainOfThought(NLPService.ExtractParameters)
            self.suggest_features = dspy.ChainOfThought(SuggestFeatures)
            
            # Add few-shot examples if training data available
            if training_examples is not None and len(training_examples) > 0:
                self._setup_few_shot_examples(training_examples)

        def _setup_few_shot_examples(self, examples: pd.DataFrame):
            """Setup few-shot examples for better classification"""
            # Sample diverse examples from each intent
            few_shot_demos = []

            if len(examples) > 0:
                # Get 3-5 examples per intent for few-shot learning
                for intent in examples['intent'].unique():
                    intent_examples = examples[examples['intent'] == intent].sample(
                        min(5, len(examples[examples['intent'] == intent]))
                    )
                    for _, row in intent_examples.iterrows():
                        few_shot_demos.append(
                            dspy.Example(
                                task=row['prompt'],
                                intent=row['intent'],
                                confidence="0.95",
                                reasoning=f"This clearly describes {row['intent'].replace('_', ' ')}"
                            ).with_inputs('task')
                        )
            if few_shot_demos:
                self.classify.demos = few_shot_demos[:15]  # Limit to top 15 diverse examples

        def forward(self, user_command, dataset_columns=""):
            # Step 1: Split into tasks
            split_result = self.split_tasks(user_command=user_command, dataset_columns=dataset_columns)
            # Parse tasks
            tasks = [t.strip() for t in split_result.tasks.split('\n') if t.strip()]
            # Handle case where splitting returns empty or just the original
            # if not tasks or (len(tasks) == 1 and len(user_command.split()) < 5):
            #     tasks = [user_command]
            # Smart fallback: if no "and" / multiple verbs detected → single task
            if not tasks or all(user_command.lower().strip() == t.lower().strip() for t in tasks):
                tasks = [user_command]
            # Step 2: Process each task
            results = []
            for task in tasks:
                # Skip empty tasks
                if not task or len(task.strip()) < 3:
                    continue
                try:
                    # Classify intent
                    intent_result = self.classify(task=task)
                    # Extract parameters

                    param_result = self.extract_params(
                        task=task,
                        dataset_columns=dataset_columns,
                        intent=intent_result.intent
                    )
                    # Parse columns from model output
                    cols = self._parse_list(param_result.columns)

                    # If no specific columns mentioned → use all dataset columns
                    if not cols and dataset_columns:
                        cols = [c.strip() for c in dataset_columns.split(",") if c.strip()]
                    results.append({
                        'task': task,
                        'intent': intent_result.intent,
                        'confidence': self._parse_confidence(intent_result.confidence),
                        'reasoning': intent_result.reasoning,
                        'columns': cols,
                        'method': param_result.method if str(param_result.method).lower() != 'none' else None,
                        'other_params': self._parse_params(param_result.other_params)
                    })
                except Exception as e:
                    st.warning(f"⚠️ Error processing task '{task}': {str(e)}")
                    continue
            return results

        def _parse_confidence(self, conf_str):
            """Parse confidence score from string"""
            try:
                match = re.search(r'0?\.\d+|\d+\.?\d*', str(conf_str))
                if match:
                    val = float(match.group())
                    # Normalize to 0-1 range
                    if val > 1:
                        val = val / 100
                    return min(max(val, 0.0), 1.0)
                return 0.85
            except:
                return 0.0

        def _parse_list(self, list_str):
            """Parse comma-separated list"""
            if str(list_str).lower() == 'none':
                return []
            items = [item.strip() for item in str(list_str).split(',') if item.strip()]
            return [item for item in items if item.lower() != 'none']

        def _parse_params(self, params_str):
            """Parse key:value parameters"""
            if str(params_str).lower() == 'none':
                return {}
            params = {}
            try:
                for pair in str(params_str).split(','):
                    if ':' in pair:
                        k, v = pair.split(':', 1)
                        params[k.strip()] = v.strip()
            except:
                pass
            return params
    class ExplainStep(dspy.Signature):
        """Ask the LM to explain why a preprocessing step ran and explain the step in detail and why this method was chosen, given metadata before/after."""
        step_name = dspy.InputField(desc="Name of the preprocessing step")
        task = dspy.InputField(desc="Original user task / intent (optional)", default="")
        metadata_before = dspy.InputField(desc="Metadata before step (JSON string)", default="")
        metadata_after = dspy.InputField(desc="Metadata after step (JSON string)", default="")
        explanation = dspy.OutputField(desc="LLM explanation for why the step was executed")

    def explain_step_llm(self,
                         step_name: str,
                         task: str = "",
                         metadata_before: Optional[Dict[str, Any]] = None,
                         metadata_after: Optional[Dict[str, Any]] = None,
                         max_tokens: int = 250) -> str:
        """
        Use DSPy / the configured LM to produce a human-readable explanation why
        the given preprocessing step executed. Includes automatic key rotation on rate limits.
        Returns the LLM's explanation string.
        """
        max_retries = _key_manager.get_total_keys_count()
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                # Ensure LM is available
                if not self.lm:
                    api_key = _key_manager.get_current_key()
                    if not api_key:
                        return "Explanation failed: No API key available."
                    lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
                    self.lm = lm

                # Prepare JSON strings for metadata fields (shorten if very large)
                mb = json.dumps(metadata_before or {}, indent=2, default=str)
                ma = json.dumps(metadata_after or {}, indent=2, default=str)
                # Optionally truncate long metadata for the prompt
                def _truncate(s, n=2000):
                    return s if len(s) <= n else (s[:n] + "\n... (truncated)")

                mb = _truncate(mb)
                ma = _truncate(ma)

                # Create a ChainOfThought for the ExplainStep signature and call it
                explain_chain = dspy.ChainOfThought(NLPService.ExplainStep)
                with dspy.context(lm=self.lm):
                    resp = explain_chain(
                        step_name=step_name,
                        task=task or "",
                        metadata_before=mb,
                        metadata_after=ma,
                    )
                # The returned object usually exposes .explanation
                explanation = getattr(resp, "explanation", None)
                if not explanation:
                    # fallback: string representation
                    explanation = str(resp)
                # small cleanup
                return explanation.strip()
                
            except Exception as e:
                error_msg = str(e).lower()
                last_error = e
                
                # Check if it's a rate limit error
                if "rate" in error_msg or "quota" in error_msg or "limit" in error_msg or "429" in error_msg:
                    available = _key_manager.get_available_keys_count()
                    print(f"[WARN] Rate limit hit during explanation. Rotating key... ({available} keys available)")
                    _key_manager.mark_key_failed()
                    
                    # If no keys available, we need to wait or fail gracefully
                    if available <= 1:
                        # Extract retry time from error if possible
                        retry_seconds = "unknown"
                        if "please try again in" in error_msg:
                            import re
                            match = re.search(r'please try again in ([^s]+s)', error_msg)
                            if match:
                                retry_seconds = match.group(1)
                        return f"Explanation failed: All API keys exhausted (TPD limit reached). Retry in {retry_seconds}. Step still executed successfully."
                    
                    try:
                        api_key = _key_manager.rotate_key()
                        os.environ["GROQ_API_KEY"] = api_key
                        lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
                        self.lm = lm
                        retry_count += 1
                        time.sleep(1)  # Brief delay before retry
                        continue
                    except RuntimeError:
                        return f"Explanation failed: All API keys exhausted. {error_msg}"
                else:
                    # Not a rate limit error - return gracefully
                    return f"Explanation failed: {e}"
        
        # All retries exhausted
        return f"Explanation failed after {max_retries} retries: {last_error}"
    # @st.cache_resource
    # def build_pipeline(self, training_data: Optional[pd.DataFrame]):
    #     """Create and cache the pipeline module."""
    #     if training_data is not None and len(training_data) > 0:
    #         return AutoPrepApp.OptimizedIntentPipeline(training_examples=training_data)
    #     return AutoPrepApp.OptimizedIntentPipeline(training_examples=None)

    def _prepare_ui_config(self):
        st.set_page_config(page_title="🧠 AutoPrepAI — DSPy Optimized", layout="centered")
        st.title("💬 AutoPrepAI — DSPy-Powered with Training Data")
        # basic guidance
        st.markdown("### 🧩 Type what you want AutoPrepAI to do")
        st.markdown(
            "- Handle missing values using median and remove duplicates by column ID\n"
            "- Detect outliers with IQR method and encode categorical variables\n"
            "- Select top 10 features for modeling and fill NaNs with mean"
        )
    def run(self, user_input: str, dataset_df: Optional[pd.DataFrame] = None, dataset_path: Optional[str] = None) -> Optional[List[str]]:
        """Headless version of runUI: perform same processing without Streamlit and return detected intents.
        Includes automatic key rotation on rate limit errors.

        Args:
            user_input: command string to process (required).
            dataset_df: optional pandas DataFrame representing the dataset (preferred).
            dataset_path: optional path to CSV/Excel dataset (used if dataset_df is None).

        Returns:
            List of detected intent names (same as runUI returns on success), or [] if nothing detected.
        """
        # 1) Ensure LM is configured with key rotation capability
        max_retries = _key_manager.get_total_keys_count()
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Configure with current key
                api_key = _key_manager.get_current_key()
                os.environ["GROQ_API_KEY"] = api_key
                lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
                self.lm = lm
                
                print(f"✅ Using API Key #{_key_manager.current_index + 1}/{_key_manager.get_total_keys_count()}")
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "quota" in error_msg or "limit" in error_msg or "429" in error_msg:
                    print(f"⚠️ Key #{_key_manager.current_index + 1} rate limited. Rotating...")
                    _key_manager.mark_key_failed()
                    try:
                        _key_manager.rotate_key()
                        retry_count += 1
                        time.sleep(1)
                        continue
                    except RuntimeError as rotate_err:
                        raise RuntimeError(str(rotate_err))
                else:
                    raise

        # 2) Load training data (headless: read from disk if not provided)
        # try:
        #     training_data = pd.read_csv(self.training_csv)
        # except Exception:
        #     training_data = None
        # self.training_data = training_data

        # 3) Build pipeline
        # self.pipeline = self.build_pipeline(self.training_data)

        # Use the class-level cached pipeline instead:
        self.pipeline = NLPService._pipeline 

        # 4) Load dataset columns (if a dataset provided)
        df = None
        if dataset_df is not None:
            df = dataset_df
        elif dataset_path:
            if dataset_path.lower().endswith(".csv"):
                df = pd.read_csv(dataset_path)
            else:
                df = pd.read_excel(dataset_path)

        columns_str = ", ".join(df.columns.tolist()) if df is not None else ""

        # 5) Validate input
        if not isinstance(user_input, str) or not user_input.strip():
            raise ValueError("user_input must be a non-empty string.")

        # 6) Run pipeline with rate limit handling
        max_pipeline_retries = _key_manager.get_total_keys_count()
        pipeline_retry = 0
        
        while pipeline_retry < max_pipeline_retries:
            try:
                with dspy.context(lm=self.lm):
                    results = self.pipeline(user_command=user_input, dataset_columns=columns_str)
                break
                
            except Exception as e:
                error_msg = str(e).lower()
                if "rate" in error_msg or "quota" in error_msg or "limit" in error_msg or "429" in error_msg:
                    print(f"⚠️ Key #{_key_manager.current_index + 1} rate limited during processing. Rotating...")
                    _key_manager.mark_key_failed()
                    try:
                        api_key = _key_manager.rotate_key()
                        os.environ["GROQ_API_KEY"] = api_key
                        lm = dspy.LM(model="groq/llama-3.3-70b-versatile", api_key=api_key, max_tokens=1000)
                        self.lm = lm
                        # Rebuild pipeline with new key
                        self.pipeline = self.build_pipeline(self.training_data)
                        pipeline_retry += 1
                        time.sleep(1)
                        continue
                    except RuntimeError as rotate_err:
                        raise RuntimeError(str(rotate_err))
                else:
                    raise RuntimeError(f"Error processing command: {e}")

        if not results:
            return []

        # 7) Produce the same "intents" list as runUI
        intents = []
        for result in results:
            temp = []
            temp.append(result.get("intent"))
            if result.get("columns"):
                temp.append(result.get("columns"))
            if result.get("method"):
                temp.append(result.get("method"))
            if result.get("other_params"):
                temp.append(result.get("other_params"))
            intents.append(temp)
        print(intents)
        return df,intents
#     def runUI(self):
#         """Public entrypoint that renders the Streamlit app and wires everything."""
#         # Setup and dependencies
#         self._prepare_ui_config()
#         try:
#             self.setup_dspy()
#             st.sidebar.success("✅ Groq AI Connected")
#         except Exception as e:
#             st.error(f"❌ Setup Error: {e}")
#             st.stop()

#         # load training data and pipeline
#         self.training_data = self.load_training_data()
#         self.pipeline = self.build_pipeline(self.training_data)
#         if self.training_data is not None:
#             with st.expander("📊 Training Data Statistics"):
#                 st.write(f"**Total training samples:** {len(self.training_data)}")
#                 st.write("**Samples per intent:**")
#                 intent_counts = self.training_data['intent'].value_counts()
#                 for intent, count in intent_counts.items():
#                     st.write(f"- {intent}: {count}")
#                 st.write("\n**Sample prompts:**")
#                 for intent in self.training_data['intent'].unique()[:3]:
#                     samples = self.training_data[self.training_data['intent'] == intent].head(2)
#                     st.write(f"\n*{intent}:*")
#                     for prompt in samples['prompt']:
#                         st.write(f"  • {prompt}")

#         # File upload
#         st.markdown("### 📂 Upload Your Dataset")
#         uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
#         df = None
#         df_columns = None
#         columns_str = ""

#         if uploaded_file is not None:
#             try:
#                 if uploaded_file.name.endswith('.csv'):
#                     df = pd.read_csv(uploaded_file)
#                 else:
#                     df = pd.read_excel(uploaded_file)
#                 df_columns = df.columns.tolist()
#                 columns_str = ", ".join(df_columns)
#                 st.success(f"✅ File uploaded! **{len(df)}** rows, **{len(df_columns)}** columns.")
#                 with st.expander("📊 Preview Dataset"):
#                     st.dataframe(df.head(10))
#                     st.write(f"**Columns:** {columns_str}")
#                     st.write(f"**Shape:** {df.shape}")
#             except Exception as e:
#                 st.error(f"❌ Error reading file: {str(e)}")
#         else:
#             st.info("💡 Upload a dataset to enable column-aware intent detection")

#         # --- Feature suggestion UI ---
#         # --- Feature suggestion UI ---
#         if df is not None:
#             st.markdown("---")
#             st.markdown("### ✨ Suggest New Features")
#             top_n = st.number_input("Number of suggestions to generate", min_value=1, max_value=20, value=5, step=1, key="suggest_top_n")
            
#             if st.button("✨ Suggest Features", key="suggest_features_btn"):
#                 if not self.pipeline:
#                     st.warning("Pipeline not ready. Please try again.")
#                 else:
#                     with st.spinner("🔎 Generating feature suggestions..."):
#                         try:
#                             sample_rows = df.head(10).to_json(orient='records')
#                             # First, try the structured helper which returns parsed suggestions
#                             try:
#                                 parsed = apply_feature_engineering_agent(dataset_columns=columns_str, sample_rows=sample_rows, top_n=int(top_n))
#                                 suggestions = []
#                                 for item in parsed:
#                                     suggestions.append(item.get('raw') or f"{item.get('name')}: {item.get('description')}" + (f" | code: {item.get('code')}" if item.get('code') else ""))
#                             except Exception as agent_exc:
#                                 # Fallback to pipeline method if the helper fails
#                                 suggest_res = self.pipeline.suggest_features(
#                                     dataset_columns=columns_str,
#                                     sample_rows=sample_rows,
#                                     top_n=str(top_n)
#                                 )
#                                 suggestions_raw = getattr(suggest_res, "suggested_features", None) or str(suggest_res)
#                                 suggestions = [s.strip() for s in str(suggestions_raw).split('\n') if s.strip()]

#                             if not suggestions:
#                                 st.warning("⚠️ No suggestions returned by the model.")
#                             else:
#                                 # Store suggestions in session state
#                                 st.session_state['feature_suggestions'] = suggestions
#                                 st.session_state['df_original'] = df.copy()  # Keep original df
                                
#                                 st.success(f"✅ {len(suggestions)} suggestions generated.")
                                
#                         except Exception as e:
#                             st.error(f"❌ Error generating suggestions: {e}")
            
#             # Display suggestions and allow selection
#             if 'feature_suggestions' in st.session_state and st.session_state['feature_suggestions']:
#                 suggestions = st.session_state['feature_suggestions']
                
#                 st.markdown("---")
#                 st.markdown("### 📋 Select Features to Apply")
                
#                 # Display suggestions with checkboxes
#                 selected_features = []
#                 for idx, suggestion in enumerate(suggestions):
#                     # Create a more readable display
#                     display_text = suggestion
#                     if '| code:' in suggestion:
#                         name_desc, code_part = suggestion.split('| code:', 1)
#                         display_text = f"**{name_desc.strip()}**\n   `Code: {code_part.strip()}`"
                    
#                     if st.checkbox(display_text, key=f"feature_{idx}", value=True):
#                         selected_features.append(suggestion)
                
#                 st.markdown("---")
                
#                 # Buttons in columns
#                 col1, col2, col3 = st.columns([1, 1, 1])
                
#                 with col1:
#                     if st.button("✅ Apply Selected Features", key="apply_features_btn", type="primary"):
#                         if not selected_features:
#                             st.warning("⚠️ Please select at least one feature to apply.")
#                         else:
#                             with st.spinner("🔧 Engineering features..."):
#                                 try:
#                                     # Use the original dataframe
#                                     df_to_engineer = st.session_state.get('df_original', df).copy()
                                    
#                                     # Convert selected features to string format
#                                     features_str = "\n".join(selected_features)
                                    
#                                     # Debug: Show what we're about to apply
#                                     with st.expander("🔍 Debug: Features to Apply", expanded=False):
#                                         st.write(f"**Number of features selected:** {len(selected_features)}")
#                                         st.write(f"**Dataframe shape:** {df_to_engineer.shape}")
#                                         st.write(f"**Dataframe columns:** {list(df_to_engineer.columns)}")
#                                         st.write("**Features string:**")
#                                         st.code(features_str)
                                    
#                                     # Apply feature engineering
#                                     df_engineered = engineer_features(df_to_engineer, features_str)
                                    
#                                     # Count new features
#                                     new_columns = [col for col in df_engineered.columns if col not in df_to_engineer.columns]
                                    
#                                     if len(new_columns) == 0:
#                                         st.warning("⚠️ No new features were added. Check the debug info below.")
#                                         st.info("**Possible issues:**")
#                                         st.write("1. The code expressions may have syntax errors")
#                                         st.write("2. Column names in the code don't match your dataframe")
#                                         st.write("3. The feature format is incorrect")
                                        
#                                         # Show a sample of what the format should be
#                                         st.code("""Expected format:
#         feature_name: description | code: df['column1'] + df['column2']
#         age_squared: Square of age | code: df['age'] ** 2
#         """)
#                                     else:
#                                         st.success(f"✅ Successfully added {len(new_columns)} new features!")
                                        
#                                         # Show new columns
#                                         st.info(f"**New columns added:** {', '.join(new_columns)}")
                                        
#                                         # Preview the engineered dataset
#                                         with st.expander("📊 Preview Updated Dataset", expanded=True):
#                                             st.dataframe(df_engineered.head(10))
#                                             st.write(f"**Original shape:** {df_to_engineer.shape}")
#                                             st.write(f"**New shape:** {df_engineered.shape}")
                                        
#                                         # Store the engineered dataframe
#                                         st.session_state['df_engineered'] = df_engineered
                                        
#                                         # Download button
#                                         csv = df_engineered.to_csv(index=False)
#                                         st.download_button(
#                                             "⬇️ Download Engineered Dataset",
#                                             csv,
#                                             "engineered_data.csv",
#                                             "text/csv",
#                                             key="download_engineered"
#                                         )
                                    
#                                 except Exception as e:
#                                     st.error(f"❌ Error applying features: {e}")
#                                     import traceback
#                                     with st.expander("🔍 Error Details", expanded=True):
#                                         st.code(traceback.format_exc())
#                                         st.write("**Selected features:**")
#                                         for i, feat in enumerate(selected_features, 1):
#                                             st.write(f"{i}. `{feat}`")
                        
#                 with col2:
#                     # Download suggestions as CSV
#                     import io, csv
#                     csv_buf = io.StringIO()
#                     writer = csv.writer(csv_buf)
#                     writer.writerow(["name", "description", "code", "raw"])
#                     for s in suggestions:
#                         name = s
#                         description = ""
#                         code = ""
#                         if ':' in s:
#                             name_part, rest = s.split(':', 1)
#                             name = name_part.strip()
#                             if '| code:' in rest:
#                                 desc_part, code_part = rest.split('| code:', 1)
#                                 description = desc_part.strip()
#                                 code = code_part.strip()
#                             else:
#                                 description = rest.strip()
#                         writer.writerow([name, description, code, s])
#                     csv_bytes = csv_buf.getvalue().encode('utf-8')
#                     st.download_button(
#                         "📥 Download Suggestions CSV", 
#                         data=csv_bytes, 
#                         file_name="feature_suggestions.csv", 
#                         mime="text/csv",
#                         key="download_suggestions"
#                     )
                
#                 with col3:
#                     if st.button("🗑️ Clear Suggestions", key="clear_suggestions_btn"):
#                         if 'feature_suggestions' in st.session_state:
#                             del st.session_state['feature_suggestions']
#                         if 'df_original' in st.session_state:
#                             del st.session_state['df_original']
#                         if 'df_engineered' in st.session_state:
#                             del st.session_state['df_engineered']
#                         st.rerun()
#         st.markdown("---")
#         user_input = st.text_area(
#             "✍️ Enter your command:",
#             height=150,
#             placeholder="e.g., handle missing values using median and remove duplicates by column ID"
#         )

#         if st.button("🔍 Understand Intents", type="primary"):
#             if not user_input.strip():
#                 st.warning("⚠️ Please enter a command first.")
#                 return
#             with st.spinner("🤔 Processing with DSPy..."):
#                 try:
#                     results = self.pipeline(user_command=user_input, dataset_columns=columns_str)
#                     if not results:
#                         st.warning("⚠️ No valid tasks detected. Try rephrasing your command.")
#                         return
#                     st.subheader("🎯 Detected Intents and Parameters")
#                     intents = []
#                     for i, result in enumerate(results, 1):
#                         temp = []
#                         if result['confidence'] >= 0.8:
#                             confidence_color = "🟢"
#                         elif result['confidence'] >= 0.6:
#                             confidence_color = "🟡"
#                         else:
#                             confidence_color = "🔴"
#                         st.markdown(f"### {confidence_color} Step {i}:")
#                         st.markdown(f"**➡️ Task:** `{result['task']}`")
#                         st.write(f"**Intent:** `{result['intent']}`")
#                         st.write(f"**Confidence:** {result['confidence']:.2%}")
#                         temp.append(result['intent'])
#                         if result.get('reasoning'):
#                             st.write(f"**Reasoning:** _{result['reasoning']}_")
#                         if result['columns']:
#                             st.write(f"**📊 Applies to columns:** {', '.join(f'`{c}`' for c in result['columns'])}")
#                             temp.append(result['columns'])
#                         else:
#                             st.write("**📊 Applies to:** all columns (no specific columns mentioned)")
#                         if result['method']:
#                             st.write(f"**⚙️ Method:** `{result['method']}`")
#                             temp.append(result['method'])
#                         if result['other_params']:
#                             st.write("**🔧 Other Parameters:**")
#                             temp.append(result['other_params'])
#                             for k, v in result['other_params'].items():
#                                 st.markdown(f"- **{k}**: `{v}`")
#                         intents.append(temp)
#                         st.markdown("---")
#                     st.success("✅ Intents detected successfully!")
#                     with st.expander("📋 Summary"):
#                         st.write(f"**Total tasks detected:** {len(results)}")
#                         st.write(f"**Average confidence:** {sum(r['confidence'] for r in results) / len(results):.2%}")
#                         intents_used = [r['intent'] for r in results]
#                         st.write(f"**Intents:** {', '.join(set(intents_used))}")
#                     if df is not None:
#                         st.markdown("### 🎬 What will happen to your data:")
#                         st.info("Preview of detected operations. Actual execution coming soon!")
#                     return intents
#                 except Exception as e:
#                     st.error(f"❌ Error processing command: {str(e)}")
#                     st.info("Try simplifying your command or check the error details above.")
#         else:
#             st.info("💡 Enter a command and click **Understand Intents**.")


# if __name__ == "__main__":
#     # When running the file directly (e.g. `streamlit run NLPService.py`),
#     # instantiate the app and launch the Streamlit UI.
#     app = NLPService()
#     try:
#         app.runUI()
#     except Exception as _err:
#         # If something goes wrong while launching the UI, print the error to the console
#         # so the user can see it in the terminal that invoked Streamlit.
#         print(f"Error launching Streamlit UI: {_err}")
#         raise