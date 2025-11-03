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
config = AutoConfig.from_pretrained("./intent_model")
tokenizer = AutoTokenizer.from_pretrained("./intent_model")
model2 = AutoModelForSequenceClassification.from_pretrained("./intent_model")
id2label = config.id2label

def classifier(text):
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model2(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return [{"label": id2label[predicted_class], "score": torch.softmax(outputs.logits, dim=1)[0][predicted_class].item()}]

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

available_methods = ["mean", "median", "mode", "average", "drop", "fill", "interpolate"]

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

    for token in doc:
        # Condition to trigger a split point
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

def extract_parameters(text):
    doc = nlp(text.lower())
    params = {}
    for i, token in enumerate(doc):
        if token.text == "column" and i + 1 < len(doc):
            params["column"] = doc[i + 1].text

    method = extract_semantic_method(text)
    if method:
        params["method"] = method
    return params


def process_intents(user_input):
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

        params = extract_parameters(clause)  # extract params from original clause (keeps column names, etc.)
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

user_input = st.text_area("✍️ Enter your command:", height=150,
                          placeholder="e.g., handle missing values using median and remove duplicates by column ID")

if st.button("🔍 Understand Intents"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a command first.")
    else:
        print("User input:", user_input)
        results, cleaned, split_type = process_intents(user_input)
        # results, cleaned = process_intents(user_input)

        st.subheader("🧹 Preprocessing Result")
        st.write(f"**Cleaned Text:** `{cleaned}`")
        st.write(f"**Chosen Splitting Strategy:** `{split_type}`")

        st.subheader("🎯 Detected Intents and Parameters")
        for i, r in enumerate(results, 1):
            st.markdown(f"### Step {i}:")
            st.markdown(f"**➡️ Clause:** `{r['clause']}`")
            # st.write(f"Predicted Intent: `{r['intent']}`")
            st.write(f"Closest Phrase: `{r['matched']}`")
            st.write(f"Similarity Score: `{r['score']:.2f}`")
            if r["params"]:
                st.write("🧩 Extracted Parameters:")
                for k, v in r["params"].items():
                    st.markdown(f"- **{k}** → `{v}`")
            else:
                st.write("No extra parameters found.")
            st.markdown("---")

        st.success("✅ Intents detected successfully!")
else:
    st.info("💡 Enter a command and click **Understand Intents**.")
