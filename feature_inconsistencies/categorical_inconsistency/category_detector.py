# detector/category_detector.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import textdistance
import numpy as np


from rapidfuzz import fuzz, process
from .constants import *
from .utils import normalize_text, looks_non_categorical

class CategoryInconsistencyDetector:
    def __init__(self, similarity_threshold: float = FUZZY_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold

    def is_categorical(self, series: pd.Series) -> bool:
        unique_ratio = series.nunique() / len(series)
        if unique_ratio > CATEGORICAL_UNIQUE_RATIO_THRESHOLD:
            return False
        if looks_non_categorical(series):
            return False
        return True

    def detect(self, df: pd.DataFrame, column: str):
        series = normalize_text(df[column])

        # Always check for invalid values
        invalids = [v for v in df[column].astype(str).str.lower().unique()
                    if v in INVALID_CATEGORY_VALUES]
        issues = []
        inconsistent_values = []
        inconsistent_indices = []

        if invalids:
            issues.append({"type": "invalid_values", "values": invalids})
            # Collect indices of invalid values
            for val in invalids:
                indices = df[df[column].astype(str).str.lower() == val].index.tolist()
                inconsistent_values.extend([val] * len(indices))
                inconsistent_indices.extend(indices)

        # Check if column is categorical
        if not self.is_categorical(series):
            return {
                "column": column,
                "is_categorical": False,
                "issues": issues,
                "inconsistent_values": inconsistent_values,
                "inconsistent_indices": inconsistent_indices,
                "message": "Column likely not categorical (too unique or pattern-based)."
                        + (" Invalid values found." if invalids else "")
            }

        # Otherwise do fuzzy detection
        # unique_vals = series.dropna().unique().tolist()
        # similar_groups = self._find_similar_groups(unique_vals)
        # if similar_groups:
        #     issues.append({"type": "similar_categories", "groups": similar_groups})
        #     # Collect indices for each similar group
        #     for group in similar_groups:
        #         for val in group:
        #             indices = df[df[column].astype(str).str.lower() == val].index.tolist()
        #             inconsistent_values.extend([val] * len(indices))
        #             inconsistent_indices.extend(indices)

        # similar_groups = self.find_similar_groups_tfidf(unique_vals)

        # if similar_groups:
        #     issues.append({"type": "tfidf_similarity", "groups": similar_groups})
        #     for group in similar_groups:
        #         for val in group:
        #             idxs = df[df[column].astype(str).str.lower() == val].index.tolist()
        #             inconsistent_values.extend([val] * len(idxs))
        #             inconsistent_indices.extend(idxs)
        
        # similar_groups = self.find_similar_groups_textdistance(unique_vals)
        # if similar_groups:
        #     issues.append({"type": "textdistance", "groups": similar_groups})
        #     for group in similar_groups:
        #         for val in group:
        #             idxs = df[df[column].astype(str).str.lower() == val].index.tolist()
        #             inconsistent_values.extend([val] * len(idxs))
        #             inconsistent_indices.extend(idxs)

        return {
            "column": column,
            "is_categorical": True,
            "issues": issues,
            "inconsistent_values": inconsistent_values,
            "inconsistent_indices": inconsistent_indices,
            "message": "Inconsistencies detected." if issues else "No issues found."
        }

    def _find_similar_groups(self, values):
        groups, visited = [], set()
        for v in values:
            if v in visited:
                continue
            matches = process.extract(v, values, scorer=fuzz.token_sort_ratio, limit=None)
            similar = [x for x, score, _ in matches if score >= self.similarity_threshold and x != v]
            if similar:
                group = [v] + similar
                groups.append(group)
                visited.update(group)
        return groups
    
    def find_similar_groups_tfidf(self, values, similarity_threshold=0.01):
        """
        values: list of strings (unique categorical values)
        similarity_threshold: float, similarity above this is considered similar
        """
        if len(values) < 2:
            return []
        
        values = list(set(values))  # remove exact duplicates

        # Character-level TF-IDF works better for short strings like categories
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        tfidf_matrix = vectorizer.fit_transform(values)

        # Cosine similarity between all vectors
        similarity_matrix = cosine_similarity(tfidf_matrix)

        groups = []
        visited = set()
        
        for i, val in enumerate(values):
            if val in visited:
                continue
            # Find all similar values above threshold
            similar = [values[j] for j, score in enumerate(similarity_matrix[i]) if score >= similarity_threshold and j != i]
            if similar:
                group = [val] + similar
                groups.append(group)
                visited.update(group)

        return groups
    
    def find_similar_groups_textdistance(self, values, similarity_threshold=0.85, metric='jaro_winkler'):
        """
        Groups similar categorical values using string similarity metrics from `textdistance`.
        
        Args:
            values (list[str]): Unique categorical values.
            similarity_threshold (float): Values with similarity >= threshold are grouped.
            metric (str): Any similarity metric from textdistance (e.g. 'jaro_winkler', 'levenshtein', 'cosine', 'jaccard').

        Returns:
            list[list[str]]: List of groups of similar values.
        """
        if len(values) < 2:
            return []

        # Get similarity function dynamically
        similarity_func = getattr(textdistance, metric)

        groups = []
        visited = set()

        for i, val in enumerate(values):
            if val in visited:
                continue

            current_group = [val]
            for other in values[i + 1:]:
                score = similarity_func.normalized_similarity(val, other)
                if score >= similarity_threshold:
                    current_group.append(other)
                    visited.add(other)

            if len(current_group) > 1:
                groups.append(current_group)
                visited.update(current_group)

        return groups
