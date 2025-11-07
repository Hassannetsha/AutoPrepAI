"""
Model management and loading functionality.
"""

import torch
import spacy
from spacy.symbols import ORTH
from sentence_transformers import SentenceTransformer
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoConfig
)
import streamlit as st

@st.cache_resource
def _load_general_models():
    """Load and cache general NLP models."""
    # Device configuration
    device = 0 if torch.cuda.is_available() else -1
    
    # Load models
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    splitter = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        device=device
    )
    zero_shot = pipeline(
        "zero-shot-classification",
        model="typeform/distilbert-base-uncased-mnli",
        device=device
    )
    
    # Configure spaCy
    nlp = spacy.load("en_core_web_sm")
    nlp.tokenizer.add_special_case("id", [{ORTH: "id"}])
    nlp.tokenizer.add_special_case("ID", [{ORTH: "ID"}])
    
    return embed_model, splitter, zero_shot, nlp

@st.cache_resource
def _load_intent_classifier():
    """Load and cache intent classification model."""
    model_path = "../intent_model"  # Path relative to the nlp directory
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        ignore_mismatched_sizes=True
    )
    model.eval()
    return tokenizer, model, config.id2label

class ModelManager:
    """Manages all ML models used in AutoPrepAI."""
    
    def __init__(self):
        """Initialize the model manager."""
        # Load models using cached functions
        self.embed_model, self.splitter, self.zero_shot, self.nlp = _load_general_models()
        self.tokenizer, self.intent_model, self.id2label = _load_intent_classifier()
    
    def classify_intent(self, text: str) -> dict:
        """
        Classify text intent using the fine-tuned model.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dict with predicted label and confidence score
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        with torch.no_grad():
            outputs = self.intent_model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1).item()
        confidence = torch.softmax(outputs.logits, dim=1)[0][predicted].item()
        return {
            "label": self.id2label[predicted],
            "score": confidence
        }