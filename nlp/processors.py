"""
Text processing and preprocessing functionality.
"""

import re
import contractions
from typing import List

class TextProcessor:
    """Handles text preprocessing and cleaning."""
    
    def __init__(self, model_manager):
        """
        Initialize text processor.
        
        Args:
            model_manager: Instance of ModelManager
        """
        self.nlp = model_manager.nlp
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize text for ML processing.
        
        Steps:
        1. Expand contractions (e.g., "don't" -> "do not")
        2. Normalize case and remove digits
        3. Lemmatize using spaCy
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned and normalized text
        """
        # Expand contractions
        text = contractions.fix(text)
        
        # Normalize text
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Process with spaCy
        doc = self.nlp(text)
        tokens = [
            token.lemma_.lower()
            for token in doc
            if not token.is_punct and not token.is_space
        ]
        return ' '.join(tokens)
    
    def split_by_dependency(self, text: str) -> List[str]:
        """
        Split text into clauses based on linguistic dependencies.
        
        Args:
            text: Input text to split
            
        Returns:
            List of clause strings
        """
        doc = self.nlp(text)
        clauses = []
        current_clause = []
        conjunctions = {"and", "or", "then", "next", "after", "finally"}
        first_verb_seen = False

        for i, token in enumerate(doc):
            if token.text.lower() in conjunctions:
                if i > 0 and i < len(doc) - 1:
                    if doc[i-1].pos_ == "NOUN" and doc[i+1].pos_ == "NOUN":
                        current_clause.append(token.text)
                        continue
            if (
                (token.pos_ == "VERB" and token.dep_ in ["ROOT", "conj"])
                or (token.text.lower() in conjunctions)
            ):
                if first_verb_seen:
                    if current_clause:
                        clauses.append(" ".join(current_clause))
                        current_clause = []
                else:
                    first_verb_seen = True
                if token.text.lower() in conjunctions:
                    continue

            current_clause.append(token.text)

        # Add last clause if anything remains
        if current_clause:
            clauses.append(" ".join(current_clause).strip())

        # Filter out empty strings
        return [c for c in clauses if c]


class IntentProcessor:
    """Handles intent processing and classification."""
    
    def __init__(self, model_manager, text_processor, parameter_extractor):
        """
        Initialize intent processor.
        
        Args:
            model_manager: Instance of ModelManager
            text_processor: Instance of TextProcessor
            parameter_extractor: Instance of ParameterExtractor
        """
        self.model_manager = model_manager
        self.text_processor = text_processor
        self.parameter_extractor = parameter_extractor
    
    def process_intents(self, user_input: str, df_columns: List[str] = None) -> tuple[List[dict], str]:
        """
        Process user input to extract intents and parameters.
        
        Steps:
        1) Split instruction into tasks using dependency parsing
        2) For each clause:
           - Preprocess (lemmatize)
           - Predict intent via classifier
           - Extract params using ML
           
        Args:
            user_input: Raw user instruction text
            df_columns: Optional list of dataset column names
            
        Returns:
            Tuple of (list of processed results, cleaned overall text)
        """
        clauses = self.text_processor.split_by_dependency(user_input)
        results = []
        
        for clause in clauses:
            cleaned = self.text_processor.preprocess_text(clause)
            intent_result = self.model_manager.classify_intent(cleaned)
            params = self.parameter_extractor.extract_parameters(clause, df_columns)
            
            results.append({
                "clause": clause,
                "cleaned_clause": cleaned,
                "matched_intent": intent_result["label"],
                "intent_score": intent_result["score"],
                "params": params
            })
            
        overall_cleaned = self.text_processor.preprocess_text(user_input)
        return results, overall_cleaned