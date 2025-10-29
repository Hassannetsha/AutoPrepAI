import streamlit as st
# import nltk
# import re
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from nltk.corpus import wordnet
# import contractions



# # =============== Setup ===============
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('averaged_perceptron_tagger_eng')

# lemmatizer = WordNetLemmatizer()
# stop_words = set(stopwords.words('english'))
# stop_words.add('ever')

# # =============== Helper functions ===============
# def expand_contractions(text):
#     return contractions.fix(text)
# def get_wordnet_pos(word):
#     tag = nltk.pos_tag([word])[0][1][0].upper()
#     tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
#     return tag_dict.get(tag, wordnet.NOUN)

# def normalize_text(text):
#     text = text.lower()
#     text = re.sub(r'\d+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# def preprocess_text(text):
#     text = expand_contractions(text)
#     text = normalize_text(text)
#     tokens = nltk.word_tokenize(text)
#     tokens = [word for word in tokens if word not in stop_words]
#     lemmatized_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in tokens]
#     return {
#         "normalized_text": text,
#         "tokens": tokens,
#         "lemmatized_tokens": lemmatized_tokens,
#         "clean_text": ' '.join(lemmatized_tokens)
#     }
import spacy
import re
import contractions

# Load once globally (important for speed)
nlp = spacy.load("en_core_web_sm")

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
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    clean_text = ' '.join(tokens)

    return {
        "normalized_text": text,
        "tokens": [token.text for token in doc],
        "lemmatized_tokens": tokens,
        "clean_text": clean_text
    }

# =============== Streamlit UI ===============
st.set_page_config(page_title="NLP Text Preprocessing", layout="wide")

st.title("🧠 NLP Text Preprocessing Demo")

st.markdown("""
### 🧩 What this does:
This web app demonstrates basic NLP preprocessing:
- **Tokenization**
- **Stopword Removal**
- **Lemmatization**
- **Text Normalization**
""")

user_input = st.text_area("✍️ Enter your text below:", height=150, placeholder="Type or paste your text here...")

if st.button("Process Text"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text first.")
    else:
        result = preprocess_text(user_input)

        st.subheader("🔹 Normalized Text")
        st.write(result["normalized_text"])

        st.subheader("🔹 Tokens (After Tokenization & Stopword Removal)")
        st.write(result["tokens"])

        st.subheader("🔹 Lemmatized Tokens")
        st.write(result["lemmatized_tokens"])

        st.subheader("✅ Final Cleaned Text")
        st.success(result["clean_text"])
else:
    st.info("💡 Enter text and click **Process Text** to see results.")
