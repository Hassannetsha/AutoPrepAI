import pandas as pd
import numpy as np

from .encoder import TextEncoder
from .index import VectorIndex
from .cross_encoder import CrossEncoderReranker

class SemanticDuplicateDetector:
    def __init__(
        self,
        encoder: TextEncoder,
        vector_index: VectorIndex,
        threshold: float = 0.70,
        cross_encoder: CrossEncoderReranker | None = None,
        cross_encoder_candidate_limit: int = 2000,
    ) -> None:
        self.encoder = encoder
        self.vector_index = vector_index
        self.threshold = threshold
        self.cross_encoder = cross_encoder
        self.cross_encoder_candidate_limit = cross_encoder_candidate_limit

    def find_duplicates(
        self,
        df: pd.DataFrame,
        text_column: str,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        t = threshold if threshold is not None else self.threshold
        texts = df[text_column].tolist()

        print("Encoding texts...")
        embeddings = self.encoder.encode(texts)

        print("Searching for nearest neighbors...")
        D, I = self.vector_index.search(embeddings)

        pairs = self._extract_pairs(D, I, t)

        if pairs.empty:
            return self._empty_pairs_df()
        
        pairs["text_1"] = df.iloc[pairs["query_index_1"].values][text_column].values
        pairs["text_2"] = df.iloc[pairs["query_index_2"].values][text_column].values

        if self.cross_encoder is not None:
            if len(pairs) <= self.cross_encoder_candidate_limit:
                pairs = self._rerank(pairs)
            else:
                print(
                    f"  Cross-encoder skipped: {len(pairs)} candidate pairs exceeds "
                    f"the limit of {self.cross_encoder_candidate_limit}. "
                    f"Running bi-encoder only. Consider raising the threshold "
                    f"to reduce candidates."
                )        
        return pairs
    
    def remove_duplicates(
            self,
            df: pd.DataFrame,
            text_column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.reset_index(drop=True)
        pairs = self.find_duplicates(df, text_column)

        if pairs.empty:
            print("No semantic duplicates found.")
            return df, self._empty_pairs_df()
        
        to_remove = set(pairs["query_index_2"].values)
        df_dedup = df.drop(index=to_remove).reset_index(drop=True)  
        print(f"Threshold {self.threshold}: found {len(to_remove)} duplicates. "
        f"{len(df)} → {len(df_dedup)} rows.")

        return df_dedup, pairs
    
    def remove_duplicates_multicolumn(
            self,
            df: pd.DataFrame,
            text_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = df.reset_index(drop=True)
        print(f"Combining columns for multi-column matching: {text_columns}")

        df_temp = df.copy()
        df_temp["_combined"] = self._combine_columns(df, text_columns)

        pairs = self.find_duplicates(df_temp, "_combined")

        if pairs.empty:
            print("No semantic duplicates found across combined columns.")
            return df, pd.DataFrame()
        
        to_remove = set(pairs["query_index_2"].values)
        df_dedup = df.drop(index=to_remove).reset_index(drop=True)
        print(f"Multi-column: found {len(to_remove)} duplicates. "
              f"{len(df)} → {len(df_dedup)} rows.")
        return df_dedup, pairs
    
    #  Private helpers                                                     #
    def _extract_pairs(
        self,
        D: np.ndarray,
        I: np.ndarray,
        threshold: float,
    ) -> pd.DataFrame:
        """
        Convert raw FAISS search results into a deduplicated pairs DataFrame.
        Only keeps pairs (i, j) where i < j to avoid duplicating each pair.
        """
        mask = D > threshold
        rows, cols = np.where(mask)
 
        # Keep only upper-triangle pairs (i < j) to avoid duplicates
        valid = rows < I[rows, cols]
        rows, cols = rows[valid], cols[valid]
 
        if len(rows) == 0:
            return pd.DataFrame()
 
        return pd.DataFrame({
            "query_index_1": rows,
            "query_index_2": I[rows, cols],
            "similarity": D[rows, cols],
        })
 
    def _rerank(self, pairs: pd.DataFrame) -> pd.DataFrame:
        if self.cross_encoder is None:
            return pairs
        
        text_pairs = list(zip(pairs["text_1"].tolist(), pairs["text_2"].tolist()))
        scores = self.cross_encoder.rerank(text_pairs)
        _, filtered_scores = self.cross_encoder.filter_pairs(text_pairs, scores)

        keep_mask = scores >= self.cross_encoder.threshold
        pairs = pairs[keep_mask].reset_index(drop=True)
        pairs["cross_encoder_score"] = filtered_scores
        return pairs
 
    @staticmethod
    def _combine_columns(df: pd.DataFrame, text_columns: list[str]) -> pd.Series:
        combined = df[text_columns[0]].astype(str)
        for col in text_columns[1:]:
            combined = combined + " | " + df[col].astype(str)
        return combined
 
    @staticmethod
    def _empty_pairs_df() -> pd.DataFrame:
        return pd.DataFrame(
            columns=["query_index_1", "query_index_2", "similarity", "text_1", "text_2"]
        )
 