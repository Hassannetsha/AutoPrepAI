import streamlit as st
import pandas as pd
# import nltk
import re
# from nltk.corpus import stopwords, wordnet
# from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import spacy
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
import torch
import contractions
import benepar

# =============== Setup ===============
st.set_page_config(page_title="🧠 AutoPrepAI — Hybrid Multi-Intent", layout="centered")
st.title("💬 AutoPrepAI — Smart Hybrid Intent Understanding")

# # --- NLTK setup ---
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords', force=True)
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')

# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))

# --- Load Models ---
@st.cache_resource
def load_models():
    nlp = spacy.load("en_core_web_md")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    splitter = pipeline("text2text-generation", model="google/flan-t5-base")
    return nlp, model, splitter

@st.cache_resource
def load_intent_classifier():
    config = AutoConfig.from_pretrained("./intent_model")
    tokenizer = AutoTokenizer.from_pretrained("./intent_model")
    model2 = AutoModelForSequenceClassification.from_pretrained(
        "./intent_model",
        ignore_mismatched_sizes=True
    )
    model2.eval()  
    id2label = config.id2label
    return tokenizer, model2, id2label

tokenizer, model2, id2label = load_intent_classifier()

def classifier(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    with torch.no_grad():  
        outputs = model2(**inputs)
    
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()
    
    return [{"label": id2label[predicted_class], "score": confidence}]
    #notes
    #pytorch pretrained model
    #model.load from pretrained model


@st.cache_resource
def load_constituency_parser():
    nlp_parser = spacy.load("en_core_web_md")
    try:
        benepar.download('benepar_en3')
    except:
        pass
    nlp_parser.add_pipe("benepar", config={"model": "benepar_en3"})
    return nlp_parser

@st.cache_data
def load_intents():
    return pd.read_csv("intents.csv")

nlp, model, splitter = load_models()
nlp_parser = load_constituency_parser()
data = load_intents()
data['embedding'] = data['prompt'].apply(lambda x: model.encode(x, convert_to_tensor=True))

available_methods = ["mean", "median", "mode", "average"]

# =============== Preprocessing ===============
def preprocess_text(text):
    # 1️⃣ Expand contractions
    text = contractions.fix(text)

    # 2️⃣ Normalize text: lowercase, remove digits & extra spaces
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 3️⃣ Process with SpaCy
    doc = nlp(text)

    # 4️⃣ Lemmatize, remove stopwords, punctuation, spaces
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_punct and not token.is_space
    ]

    clean_text = ' '.join(tokens)
    # # 5️⃣ Identify subtask-like phrases (split by verbs or 'and/then')
    # clauses = []
    # current = []
    # for token in doc:
    #     if token.pos_ == "CCONJ" or token.text in ["and", "then","or"]:
    #         if current:
    #             clauses.append(" ".join(current).strip())
    #             current = []
    #     current.append(token.text)
    # if current:
    #     clauses.append(" ".join(current).strip())

    # # fallback if nothing split
    # if len(clauses) == 0:
    #     clauses = [text]

    return {
        "normalized_text": text,
        "tokens": [token.text for token in doc],
        "lemmatized_tokens": tokens,
        "clean_text": clean_text,
        # "clauses": clauses
    }

# =============== Split Methods ===============
# Dependency-based
def split_by_dependency(text):
    doc = nlp(text)
    clauses = []
    current_clause = []
    conjunctions = {"and", "or", "then", "next", "after", "finally"}
    first_verb_seen = False

    for i, token in enumerate(doc):
        # Condition to trigger a split point
        if token.text.lower() in conjunctions:
        # 👇 NEW CHECK: if 'and' is surrounded by nouns (likely column names), skip splitting
            if i > 0 and i < len(doc) - 1:
                if doc[i-1].pos_ == "NOUN" and doc[i+1].pos_ == "NOUN":
                    current_clause.append(token.text)
                    continue
        if (
            (token.pos_ == "VERB" and token.dep_ in ["ROOT", "conj"])
            or (token.text.lower() in conjunctions)
        ):
            # Save what we collected so far (if not empty)
            # Skip splitting at the very first verb (to keep "Handle missing values" together)
            if first_verb_seen:
                if current_clause:
                    clauses.append(" ".join(current_clause))
                    current_clause = []
            else:
                first_verb_seen = True
            if token.text.lower() in conjunctions:
                continue

        # Add token to the current clause
        current_clause.append(token.text)

    # Add last clause if anything remains
    if current_clause:
        clauses.append(" ".join(current_clause).strip())

    # Filter out empty strings and return
    return [c for c in clauses if c]


# Transformer (Flan-T5)
def split_by_transformer(text):
    prompt = f"Split this user instruction into separate clear tasks: {text}"
    result = splitter(prompt, max_new_tokens=100)
    lines = result[0]['generated_text'].split("\n")
    return [line.strip("- ").strip() for line in lines if line.strip()]

# Constituency-based
def split_by_constituency(text):
    """Use constituency parse tree to split by verb phrases (VP)."""
    doc = nlp_parser(text)
    clauses = []
    for sent in doc.sents:
        if hasattr(sent._, "parse_tree"):
            tree = sent._.parse_tree
            for subtree in tree.subtrees(filter=lambda t: t.label() == "VP"):
                phrase = " ".join(subtree.leaves())
                if len(phrase.split()) > 2:
                    clauses.append(phrase)
    return clauses or [text]

# Smart Hybrid Decision
def smart_split(text):
    num_words = len(text.split())
    # num_verbs = len([t for t in nlp(text) if t.pos_ == "VERB"])
    
    # if num_words <= 10:
    return "dependency", split_by_dependency(text)
    # else:
    #     return "transformer", split_by_transformer(text)


# =============== Intent + Parameter Extraction ===============
def predict_intent(text):
    result = classifier(text)[0]
    return result["label"], result["score"]
    # user_emb = model.encode(text, convert_to_tensor=True)
    # sims = [float(util.cos_sim(user_emb, emb)) for emb in data['embedding']]
    # best_match = data.iloc[sims.index(max(sims))]
    # return best_match['intent'], best_match['prompt'], max(sims)
#Bert for semantic duplicate
def extract_semantic_method(text):
    text_emb = model.encode(text.split(), convert_to_tensor=True)
    best_method, best_score = None, 0
    for m in available_methods:
        method_emb = model.encode(m, convert_to_tensor=True)
        score = float(util.cos_sim(text_emb, method_emb).max())
        if score > best_score:
            best_score, best_method = score, m
    if best_score > 0.6:
        return best_method
    return None

def extract_parameters(text, df_columns=None):
    doc = nlp(text.lower())
    params = {}
    for i, token in enumerate(doc):
        if token.text in ["column", "columns"] and i + 1 < len(doc):
            next_token = doc[i + 1]
            if not next_token.is_punct and not next_token.is_space:
                if "column" not in params:
                    params["column"] = []
                params["column"].append(next_token.text)
    
    # Pattern 2: Match with DataFrame columns
    if df_columns:
        # Try advanced list extraction first
        matched_list = extract_column_list_from_text(text, df_columns)
        if matched_list:
            if "column" not in params:
                params["column"] = matched_list
            else:
                # Merge with existing
                params["column"] = list(set(params["column"] + matched_list))
        else:
            # Fallback to general matching
            matched_cols = match_columns(text, df_columns)
            if matched_cols:
                if "column" not in params:
                    params["column"] = matched_cols
                else:
                    params["column"] = list(set(params["column"] + matched_cols))
    
    # Convert single column list to string for consistency
    if "column" in params and isinstance(params["column"], list):
        if len(params["column"]) == 1:
            params["column"] = params["column"][0]
    
    method = extract_semantic_method(text)
    if method:
        params["method"] = method
    return params

def match_columns(text, df_columns):
    """
    Match user input to actual column names in the DataFrame.
    Returns a LIST of matched columns.
    """
    if not df_columns:
        return None
    
    text_lower = text.lower()
    matched = []
    
    # Method 1: Direct exact match
    for col in df_columns:
        if col.lower() in text_lower:
            matched.append(col)
    
    if matched:
        return matched
    
    # Method 2: Normalized matching (remove spaces/underscores/hyphens)
    text_normalized = re.sub(r'[_\s-]+', '', text_lower)
    
    for col in df_columns:
        col_normalized = re.sub(r'[_\s-]+', '', col.lower())
        
        if col_normalized in text_normalized or text_normalized in col_normalized:
            matched.append(col)    
    if matched:
        return matched
    
    # Method 3: Word overlap matching (SPLIT by underscores too!)
    # Split by spaces, underscores, and hyphens
    text_words = set(re.split(r'[_\s-]+', text_lower))
    text_words = {w for w in text_words if w}  # Remove empty strings    
    for col in df_columns:
        # Split column by underscores, spaces, hyphens
        col_words = set(re.split(r'[_\s-]+', col.lower()))
        col_words = {w for w in col_words if w}  # Remove empty strings        
        if col_words and text_words:
            overlap = text_words & col_words
            score = len(overlap) / len(col_words)            
            if score >= 0.8:
                matched.append(col)
    
    return matched if matched else None

def extract_column_list_from_text(text, df_columns):
    """
    Advanced extraction for comma/and-separated column lists.
    Examples:
    - "age, salary, and experience" -> ["age", "salary", "experience"]
    - "user_id and customer_id" -> ["user_id", "customer_id"]
    """
    # Split by common delimiters
    potential_cols = re.split(r'[,;&]|\band\b|\bor\b', text.lower())
    potential_cols = [col.strip() for col in potential_cols]
    
    matched = []
    for potential in potential_cols:
        if not potential:
            continue
        
        # Try to match each potential column
        for col in df_columns:
            col_lower = col.lower()
            potential_normalized = re.sub(r'[_\s-]+', '', potential)
            col_normalized = re.sub(r'[_\s-]+', '', col_lower)
            
            if (col_lower == potential or 
                col_lower in potential or 
                potential in col_lower or
                col_normalized == potential_normalized):
                if col not in matched:
                    matched.append(col)
                break
    
    return matched if matched else None


def process_intents(user_input, df_columns=None):
    # Use original text for splitting so we don't lose conjunctions
    split_type, clauses = smart_split(user_input)
    # preprocessed = preprocess_text(user_input)
    # clauses = preprocessed["clauses"]

    results = []
    for clause in clauses:
        # preprocess each clause BEFORE predicting intent (clean for embeddings)
        cleaned_clause = preprocess_text(clause)
        # intent, matched, score = predict_intent(cleaned_clause["clean_text"])
        matched, score = predict_intent(cleaned_clause["clean_text"])
        # matched, score = predict_intent(clause)

        params = extract_parameters(clause,df_columns)  # extract params from original clause (keeps column names, etc.)
        results.append({
            "clause": clause,
            "cleaned_clause": cleaned_clause,
            # "intent": intent,
            "params": params,
            "matched": matched,
            "score": score
        })
    # also return the overall cleaned text if you need it
    overall_cleaned = preprocess_text(user_input)
    return results, overall_cleaned, split_type
    # return results, overall_cleaned


# =============== Streamlit UI ===============
st.markdown("""
### 🧩 Type what you want AutoPrepAI to do  
Examples:
- "Handle missing values using median and remove duplicates by column ID"
- "Normalize data, then detect outliers and drop extreme rows"
""")

# ========== Add File Upload ==========
st.markdown("### 📂 Upload Your Dataset ")
uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])

df = None
df_columns = None

if uploaded_file is not None:
    try:
        # Read file based on type
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        df_columns = df.columns.tolist()
        
        st.success(f"✅ File uploaded successfully! Found **{len(df)}** rows and **{len(df_columns)}** columns.")
        
        # Show dataset info
        with st.expander("📊 Preview Dataset"):
            st.dataframe(df.head(10))
            st.write(f"**Columns:** {', '.join(df_columns)}")
            st.write(f"**Shape:** {df.shape}")
            
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
else:
    st.info("💡 Upload a dataset to enable column-aware intent detection")

st.markdown("---")

# ========== Text Input ==========
user_input = st.text_area("✍️ Enter your command:", height=150,
                          placeholder="e.g., handle missing values using median and remove duplicates by column ID")

if st.button("🔍 Understand Intents"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a command first.")
    else:
        print("User input:", user_input)
        
        # Pass columns to the function (or None if no file uploaded)
        results, cleaned, split_type = process_intents(user_input, df_columns)

        st.subheader("🧹 Preprocessing Result")
        st.write(f"**Cleaned Text:** `{cleaned}`")
        st.write(f"**Chosen Splitting Strategy:** `{split_type}`")

        st.subheader("🎯 Detected Intents and Parameters")
        for i, r in enumerate(results, 1):
            st.markdown(f"### Step {i}:")
            st.markdown(f"**➡️ Clause:** `{r['clause']}`")
            st.write(f"**Closest Phrase:** `{r['matched']}`")
            st.write(f"**Similarity Score:** `{r['score']:.2f}`")
            if r["params"]:
                st.write("🧩 Extracted Parameters:")
                for k, v in r["params"].items():
                    st.markdown(f"- **{k}** → `{v}`")
            else:
                st.write("No extra parameters found.")
            st.markdown("---")

        st.success("✅ Intents detected successfully!")
        
        # If dataset is uploaded, show preview of expected results
        if df is not None:
            st.markdown("### 🎬 What will happen to your data:")
            st.info("This is a preview of the detected operations. Actual execution coming soon!")
            
else:
    st.info("💡 Enter a command and click **Understand Intents**.")