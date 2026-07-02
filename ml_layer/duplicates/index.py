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
        self.k_neighbors = k_neighbors # number of nearest neighbors to return
        self.n_clusters = n_clusters # number of clusters (faster search)
        self.nprobe = nprobe # number of clusters to search

    def build(self, embeddings: np.ndarray) -> faiss.IndexIVFFlat:
        n_points, d = embeddings.shape
        n_clusters = min(self.n_clusters, n_points) # if embedddings are less than n_clusters, use n_points as n_clusters

        quantizer = faiss.IndexFlatIP(d) # Assigns vectors to the nearest cluster
        
        index = faiss.IndexIVFFlat(
            quantizer, 
            d, 
            n_clusters, 
            faiss.METRIC_INNER_PRODUCT # cosine similarity (after L2 normalization)
        )

        index.train(embeddings) # learn cluster centroids
        index.add(embeddings) # store embeddings in the index
        return index

    def search(self, embeddings: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        index = self.build(embeddings) # build searchable index
        index.nprobe = self.nprobe # search this many nearby clusters
        return index.search(embeddings, self.k_neighbors) # returns (similarities, indices)