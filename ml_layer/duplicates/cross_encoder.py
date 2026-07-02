from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

class CrossEncoderReranker:
    # needed for false positives filtering
    def __init__(
            self,
            model_name: str = "cross-encoder/stsb-roberta-base",
            threshold: float = 0.5,
            batch_size: int = 64,
    ):
        self.threshold = threshold # minimum similarity to keep a pair
        self.batch_size = batch_size # number of pairs to process at once
        self.model = CrossEncoder(model_name)

    def rerank(
            self,
            pairs: list[tuple[str, str]],
    ) -> np.ndarray:
        if not pairs:
            return np.array([])
        print(f"  Cross-encoder reranking {len(pairs)} candidate pairs...")

        scores = self.model.predict(pairs, batch_size=self.batch_size) # compute similarity scores for each pair
        return np.array(scores)
    
    def filter_pairs(
            self,
            texts: list[tuple[str, str]],
            scores: np.ndarray
    ) -> tuple[list[tuple[str, str]], np.ndarray]:
        mask = scores >= self.threshold # keep only high-confidence matches
        filtered_texts = [t for t, keep in zip(texts, mask) if keep]
        filtered_scores = scores[mask]
        rejected = len(texts) - len(filtered_texts)
        if rejected > 0:
            print(f"  Cross-encoder rejected {rejected} false positives.")
        return filtered_texts, filtered_scores    