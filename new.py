import streamlit as st
import pandas as pd
import dspy
from typing import List, Optional
import re
import os
os.environ['GROQ_API_KEY'] = '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'

# =============== DSPy Setup ===============
st.set_page_config(page_title="🧠 AutoPrepAI — DSPy Optimized", layout="centered")
st.title("💬 AutoPrepAI — DSPy-Powered with Training Data")

# Initialize DSPy with Groq
@st.cache_resource
def setup_dspy():
    api_key = os.getenv('GROQ_API_KEY')
    
    if not api_key:
        st.warning("⚠️ GROQ_API_KEY not found!")
        st.info("""
        **Quick Setup:**
        1. Get free API key: https://console.groq.com/
        2. Enter it below:
        """)
        api_key = st.text_input("Enter Groq API Key:", type="password", key="api_key_input")
        if api_key:
            os.environ['GROQ_API_KEY'] = api_key
        else:
            st.stop()
    
    lm = dspy.LM(
        model='groq/llama-3.3-70b-versatile',
        api_key=api_key,
        max_tokens=1000
    )
    
    dspy.settings.configure(lm=lm)
    return lm

try:
    lm = setup_dspy()
    st.sidebar.success("✅ Groq AI Connected")
except Exception as e:
    st.error(f"❌ Setup Error: {e}")
    st.stop()

# =============== Load Training Data ===============
@st.cache_data
def load_training_data():
    """Load your augmented training data"""
    try:
        df = pd.read_csv("intents_augmented.csv")
        return df
    except FileNotFoundError:
        st.warning("⚠️ Training file 'intents_augmented.csv' not found. Using basic mode.")
        return None

training_data = load_training_data()

# =============== DSPy Signatures ===============

# class SplitIntoTasks(dspy.Signature):
#     """Split a complex user command into individual, atomic tasks.
#     Each task should be a complete, independent operation."""
    
#     user_command = dspy.InputField(desc="The complete user command with possibly multiple tasks")
#     dataset_columns = dspy.InputField(desc="Available column names in the dataset (comma-separated)", default="")
    
#     tasks = dspy.OutputField(desc="List of individual tasks, one per line. Each task should be a complete sentence.")
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
    
    tasks = dspy.OutputField(desc="Each preprocessing action on a separate line, but not per column unless explicitly stated.")

class ClassifyIntent(dspy.Signature):
    """Classify the data preprocessing intent from a task description.
    
    Available intents:
    - handle_missing_values: Fill, impute, or handle missing/null/NaN values
    - detect_outliers: Identify and remove outliers, anomalies, or extreme values
    - keep_outliers: Preserve or keep outliers in the data
    - remove_duplicates: Remove duplicate rows or entries
    - encode_categorical: Convert categorical/text columns to numeric
    - feature_selection: Select important features or columns for modeling
    """
    
    task = dspy.InputField(desc="A single preprocessing task description")
    
    intent = dspy.OutputField(desc="The intent category (must be one of: handle_missing_values, detect_outliers, keep_outliers, remove_duplicates, encode_categorical, feature_selection)")
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

# =============== DSPy Pipeline with Few-Shot Examples ===============

class OptimizedIntentPipeline(dspy.Module):
    """Intent understanding pipeline with few-shot learning from training data"""
    
    def __init__(self, training_examples=None):
        super().__init__()
        self.split_tasks = dspy.ChainOfThought(SplitIntoTasks)
        self.classify = dspy.ChainOfThought(ClassifyIntent)
        self.extract_params = dspy.ChainOfThought(ExtractParameters)
        
        # Add few-shot examples if training data available
        if training_examples is not None and len(training_examples) > 0:
            self._setup_few_shot_examples(training_examples)
    
    def _setup_few_shot_examples(self, examples):
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
        
        # Add examples to classifier
        if few_shot_demos:
            self.classify.demos = few_shot_demos[:15]  # Limit to top 15 diverse examples
            st.sidebar.info(f"📚 Using {len(self.classify.demos)} training examples")
    
    def forward(self, user_command, dataset_columns=""):
        # Step 1: Split into tasks
        split_result = self.split_tasks(
            user_command=user_command,
            dataset_columns=dataset_columns
        )
        
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
                    'method': param_result.method if param_result.method.lower() != 'none' else None,
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

# =============== Load Pipeline ===============
@st.cache_resource
def load_pipeline(_training_data):
    """Load and cache the optimized pipeline"""
    if _training_data is not None and len(_training_data) > 0:
        st.sidebar.info(f"🎓 Training with {len(_training_data)} examples")
        return OptimizedIntentPipeline(training_examples=_training_data)
    else:
        st.sidebar.warning("⚠️ No training data - using zero-shot mode")
        return OptimizedIntentPipeline(training_examples=None)

pipeline = load_pipeline(training_data)

# =============== Streamlit UI ===============
st.markdown("""
### 🧩 Type what you want AutoPrepAI to do  
Examples:
- "Handle missing values using median and remove duplicates by column ID"
- "Detect outliers with IQR method and encode categorical variables"
- "Select top 10 features for modeling and fill NaNs with mean"
""")

# Show training stats
if training_data is not None:
    with st.expander("📊 Training Data Statistics"):
        st.write(f"**Total training samples:** {len(training_data)}")
        st.write("**Samples per intent:**")
        intent_counts = training_data['intent'].value_counts()
        for intent, count in intent_counts.items():
            st.write(f"- {intent}: {count}")
        
        st.write("\n**Sample prompts:**")
        for intent in training_data['intent'].unique()[:3]:
            samples = training_data[training_data['intent'] == intent].head(2)
            st.write(f"\n*{intent}:*")
            for prompt in samples['prompt']:
                st.write(f"  • {prompt}")

# ========== File Upload ==========
st.markdown("### 📂 Upload Your Dataset")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

df = None
df_columns = None
columns_str = ""

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        df_columns = df.columns.tolist()
        columns_str = ", ".join(df_columns)
        
        st.success(f"✅ File uploaded! **{len(df)}** rows, **{len(df_columns)}** columns.")
        
        with st.expander("📊 Preview Dataset"):
            st.dataframe(df.head(10))
            st.write(f"**Columns:** {columns_str}")
            st.write(f"**Shape:** {df.shape}")
            
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
else:
    st.info("💡 Upload a dataset to enable column-aware intent detection")

st.markdown("---")

# ========== Text Input ==========
user_input = st.text_area(
    "✍️ Enter your command:", 
    height=150,
    placeholder="e.g., handle missing values using median and remove duplicates by column ID"
)

if st.button("🔍 Understand Intents", type="primary"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a command first.")
    else:
        with st.spinner("🤔 Processing with DSPy..."):
            try:
                results = pipeline(
                    user_command=user_input,
                    dataset_columns=columns_str
                )
                
                if not results:
                    st.warning("⚠️ No valid tasks detected. Try rephrasing your command.")
                else:
                    st.subheader("🎯 Detected Intents and Parameters")
                    
                    for i, result in enumerate(results, 1):
                        # Color code by confidence
                        if result['confidence'] >= 0.8:
                            confidence_color = "🟢"
                        elif result['confidence'] >= 0.6:
                            confidence_color = "🟡"
                        else:
                            confidence_color = "🔴"
                        
                        st.markdown(f"### {confidence_color} Step {i}:")
                        st.markdown(f"**➡️ Task:** `{result['task']}`")
                        st.write(f"**Intent:** `{result['intent']}`")
                        st.write(f"**Confidence:** {result['confidence']:.2%}")
                        
                        if result.get('reasoning'):
                            st.write(f"**Reasoning:** _{result['reasoning']}_")
                        
                        if result['columns']:
                            st.write(f"**📊 Applies to columns:** {', '.join(f'`{c}`' for c in result['columns'])}")
                        else:
                            st.write("**📊 Applies to:** all columns (no specific columns mentioned)")

                        
                        if result['method']:
                            st.write(f"**⚙️ Method:** `{result['method']}`")
                        
                        if result['other_params']:
                            st.write("**🔧 Other Parameters:**")
                            for k, v in result['other_params'].items():
                                st.markdown(f"- **{k}**: `{v}`")
                        
                        st.markdown("---")
                    
                    st.success("✅ Intents detected successfully!")
                    
                    # Show summary
                    with st.expander("📋 Summary"):
                        st.write(f"**Total tasks detected:** {len(results)}")
                        st.write(f"**Average confidence:** {sum(r['confidence'] for r in results) / len(results):.2%}")
                        intents_used = [r['intent'] for r in results]
                        st.write(f"**Intents:** {', '.join(set(intents_used))}")
                    
                    if df is not None:
                        st.markdown("### 🎬 What will happen to your data:")
                        st.info("Preview of detected operations. Actual execution coming soon!")
                        
            except Exception as e:
                st.error(f"❌ Error processing command: {str(e)}")
                st.info("Try simplifying your command or check the error details above.")

else:
    st.info("💡 Enter a command and click **Understand Intents**.")