"""
Parameter extraction functionality.
"""

from typing import List, Dict, Optional
from sentence_transformers import util

class ParameterExtractor:
    """Handles extraction of parameters from text using ML techniques."""
    
    def __init__(self, model_manager):
        """
        Initialize parameter extractor.
        
        Args:
            model_manager: Instance of ModelManager
        """
        self.zero_shot = model_manager.zero_shot
        self.embed_model = model_manager.embed_model
        self.nlp = model_manager.nlp
        
    def extract_parameters(self, text: str, df_columns: List[str] = None) -> Dict:
        """
        Extract all parameters from text using ML techniques.
        
        Args:
            text: Input text to analyze
            df_columns: Optional list of available column names
            
        Returns:
            Dictionary of extracted parameters
        """
        params = {}
        
        # Extract method (e.g., mean, median)
        method = self.extract_semantic_method(text)
        if method:
            params["method"] = method
            
        # Extract scope and columns
        scope_info = self.detect_column_scope(text)
        params["apply_to"] = scope_info["scope"]
        
        # Extract columns
        columns = self.extract_columns_semantic(text, df_columns)
        if columns:
            if scope_info["scope"] == "all columns except":
                params["exclude"] = columns
            else:
                params["column"] = columns if len(columns) > 1 else columns[0]
                
        # Set ALL for global scope
        if scope_info["scope"] == "all columns":
            params["column"] = "ALL"
            
        return params
    
    def extract_semantic_method(
        self,
        text: str,
        candidates: List[str] = None,
        threshold: float = 0.5
    ) -> Optional[str]:
        """
        Extract statistical method using zero-shot classification.
        
        Args:
            text: Input text to analyze
            candidates: List of possible methods to consider
            threshold: Minimum confidence threshold
            
        Returns:
            Extracted method name or None if no match found
        """
        if not candidates:
            candidates = ["mean", "median", "mode", "average"]
            
        try:
            res = self.zero_shot(text, candidate_labels=candidates, multi_label=False)
            if res and res.get("scores"):
                if res["scores"][0] >= threshold:
                    return res["labels"][0]
        except Exception:
            pass
        return None
    
    def extract_columns_semantic(
        self,
        text: str,
        df_columns: List[str] = None,
        threshold: float = 0.55
    ) -> Optional[List[str]]:
        """
        Extract column references using semantic similarity.
        
        Args:
            text: Input text to analyze
            df_columns: List of available column names
            threshold: Minimum similarity threshold
            
        Returns:
            List of matched column names or None
        """
        if not df_columns:
            return None
            
        try:
            # Extract potential column tokens
            tokens = [t.text.lower() for t in self.nlp(text) if t.is_alpha]
            matched = []
            
            # Compare each token with column names
            for token in tokens:
                token_vec = self.embed_model.encode(token, convert_to_tensor=True)
                for col in df_columns:
                    col_vec = self.embed_model.encode(col, convert_to_tensor=True)
                    score = float(util.cos_sim(token_vec, col_vec))
                    if score >= threshold:
                        matched.append(col)
                        break  # Avoid double-matching
                        
            matched = list(set(matched))  # Remove duplicates
            return matched if matched else None
            
        except Exception as e:
            print(f"Error in semantic column extraction: {e}")
            return None
    
    def detect_column_scope(self, text: str,threshold: float = 0.45) -> Dict[str, str]:
        """
        Detect the scope of column references in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dict with scope information
        """
        candidate_scopes = ["None", "all columns", "all columns except", "specific columns"]
        try:
            res = self.zero_shot(text, candidate_labels=candidate_scopes, multi_label=False)

            if not res or "labels" not in res or "scores" not in res: 
                return {"scope": "all columns"}  # default changed here

            label = res["labels"][0]
            score = res["scores"][0]

            # Optional confidence fallback
            if score < threshold:
                label = "all columns"  # default fallback here too
                
            if label == "None":
                return {"scope": "all columns"}

            return {"scope": label}

        except Exception:
            return {"scope": "all columns"}  # and here
