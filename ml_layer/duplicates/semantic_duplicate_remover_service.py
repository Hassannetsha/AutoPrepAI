import pandas as pd 

from .detector import SemanticDuplicateDetector
from .encoder import TextEncoder
from .evaluator import Evaluator
from .index import VectorIndex
from .cross_encoder import CrossEncoderReranker


class SemanticDuplicateRemoverService:
    def __init__(
            self,
            model_name: str = "paraphrase-MiniLM-L6-v2",
            threshold: float = 0.85,
            k_neighbors: int = 10,
            batch_size: int = 512,
            n_clusters: int = 100,
            nprobe: int = 10,
            use_cross_encoder: bool = True,
            cross_encoder_model: str = "cross-encoder/stsb-roberta-base",
            cross_encoder_threshold: float = 0.5,
            cross_encoder_batch_size: int = 64,
            cross_encoder_candidate_limit: int = 2000,
    ) -> None:
        encoder = TextEncoder(model_name=model_name, batch_size=batch_size)
        vector_index = VectorIndex(k_neighbors=k_neighbors, n_clusters=n_clusters, nprobe=nprobe)

        cross_encoder = None
        if use_cross_encoder:
            cross_encoder = CrossEncoderReranker(
                    model_name=cross_encoder_model,
                    threshold=cross_encoder_threshold,
                    batch_size=cross_encoder_batch_size,
            )

        self.detector = SemanticDuplicateDetector(
            encoder=encoder,
            vector_index=vector_index,
            threshold=threshold,
            cross_encoder=cross_encoder,
            cross_encoder_candidate_limit=cross_encoder_candidate_limit,
        )
        self.evaluator = Evaluator(detector=self.detector)

    def remove_duplicates(
        self,
        df: pd.DataFrame,
        text_column: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
       return self.detector.remove_duplicates(df, text_column)
    
    def remove_duplicates_multicolumn(
            self,
            df: pd.DataFrame,
            text_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.detector.remove_duplicates_multicolumn(df, text_columns)
    
    def threshold_sensitivity_analysis(
            self,
            df: pd.DataFrame,
            text_column: str,
            ground_truth_pairs: list[tuple[int, int]],
            thresholds: list[float]
    ) -> pd.DataFrame:
        return self.evaluator.threshold_sensitivity_analysis(
            df, text_column, thresholds
        )
    
    def fuzzy_baseline(
            self,
            df: pd.DataFrame,
            text_column: str,
            fuzzy_threshold: float = 0.90,
    ) -> pd.DataFrame:
        return self.evaluator.fuzzy_baseline(df, text_column, fuzzy_threshold)
    
    def error_analysis(
        self,
        pairs: pd.DataFrame,
        ground_truth_pairs: list[tuple[int, int]],
        df: pd.DataFrame,
        text_column: str,
        n_samples: int = 10,
    ) -> dict:
        return self.evaluator.error_analysis(
            pairs, ground_truth_pairs, df, text_column, n_samples
        )