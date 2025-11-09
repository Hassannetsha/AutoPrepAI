import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import torch


class SemanticDuplicateRemover:
    def __init__(
        self,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        threshold: float = 0.8,
        k_neighbors: int = 10,
        batch_size: int = 512
    ):
        """
        Initialize the semantic duplicate remover.
        Args:
            model_name: HuggingFace SentenceTransformer model name.
            threshold: Cosine similarity threshold for duplicate detection.
            k_neighbors: Number of nearest neighbors to search for each record.
            batch_size: Batch size for model encoding.
        """
        self.model_name = model_name
        self.threshold = threshold
        self.k_neighbors = k_neighbors
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        num_cpu_cores = os.cpu_count()
        num_workers_to_use = max(1, num_cpu_cores - 1)
        print(f"Using {num_workers_to_use} workers for encoding.")

        with torch.no_grad():
            embeddings = self.model.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=self.batch_size
            )

        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)
        return embeddings
    
    def _build_faiss_index(self, embeddings: np.ndarray):
        d = embeddings.shape[1]
        n_points = embeddings.shape[0]
        n_clusters = min(100, n_points)
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)

        index.train(embeddings)
        index.add(embeddings)
        return index


    def _find_duplicates(self, embeddings: np.ndarray, df: pd.DataFrame, text_column: str):
        d = embeddings.shape[1]
        N = embeddings.shape[0]

        index = self._build_faiss_index(embeddings)
        index.nprobe = 10  # number of clusters to search
        D, I = index.search(embeddings, self.k_neighbors)

        # Mask to keep pairs above threshold
        mask = D > self.threshold
        rows, cols = np.where(mask)

        # Keep only unique pairs where i < j to avoid duplicates
        valid = rows < I[rows, cols]
        rows, cols = rows[valid], cols[valid]

        duplicates_df = pd.DataFrame({
            'query_index_1': rows,
            'query_index_2': I[rows, cols],
            'similarity': D[rows, cols],
        })
        duplicates_df['text_1'] = df.iloc[duplicates_df['query_index_1']][text_column].values
        duplicates_df['text_2'] = df.iloc[duplicates_df['query_index_2']][text_column].values
        return duplicates_df


    def remove_duplicates(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """
        Remove semantic duplicates from a dataframe.
        Args:
            df: Input DataFrame containing text data.
            text_column: Column name containing text to compare.
        Returns:
            A deduplicated DataFrame.
        """
        print("Encoding texts...")
        embeddings = self._encode(df[text_column].tolist())
        print("Finding semantic duplicates...")
        duplicates_df = self._find_duplicates(embeddings, df, text_column)

        if duplicates_df.empty:
            print("No semantic duplicates found.")
            return df, pd.DataFrame(columns=['query_index_1', 'query_index_2', 'similarity', 'text_1', 'text_2'])

        to_remove = set(duplicates_df['query_index_2'].values)
        df_dedup = df.drop(index=to_remove).reset_index(drop=True)

        print(f"\nTotal semantic duplicates found above {self.threshold}: {len(to_remove)}")
        print(f"Original Length: {len(df)} → Remaining: {len(df_dedup)}")

        return df_dedup, duplicates_df
