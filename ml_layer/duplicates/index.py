import numpy as np
import faiss


class VectorIndex:
    """
    Builds and searches a FAISS IVFFlat index over L2-normalized embeddings

    - builds the IVFFlat index (train + add)
    - runs nearest neighbor search and returning (distances, indices)
    - owns IVFFlat tuning parameters: n_clusters, nprobe
    """
    def __init__(self, 
                 k_neighbors: int = 10,
                 n_clusters: int = 100,
                 nprobe: int = 10):
        self.k_neighbors = k_neighbors
        self.n_clusters = n_clusters
        self.nprobe = nprobe

    def build(self, embeddings: np.ndarray) -> faiss.IndexIVFFlat:
        n_points, d = embeddings.shape
        n_clusters = min(self.n_clusters, n_points)

        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(quantizer, d, n_clusters, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        return index

    def search(self, embeddings: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        index = self.build(embeddings)
        index.nprobe = self.nprobe
        return index.search(embeddings, self.k_neighbors)