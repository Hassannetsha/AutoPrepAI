from sentence_transformers import SentenceTransformer
import numpy as np
import os 
import torch
import faiss

class TextEncoder:
    def __init__(self,
                 model_name: str ="paraphrase-MiniLM-L6-v2",
                 batch_size: int = 512,) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Encodes a list of texts into L2-normalized float32 embeddings
        suitable for FAISS inner-product search.

        - loads the SentenceTransformer model
        - batched encoding with dedupliation to avoid redundant work
        - L2 normalization (converts cosine similarity to inner product)
        """
        unique_texts = list(dict.fromkeys(texts))
        print(f"Encoding {len(unique_texts)} unique texts for {len(texts)} pairs ...")
        
        with torch.no_grad():
            embeddings = self.model.encode(
                unique_texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                batch_size=self.batch_size
            )
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        # Map back to original order
        index_map = {text: i for i, text in enumerate(unique_texts)}
        embeddings = np.stack([embeddings[index_map[text]] for text in texts])
        
        return embeddings