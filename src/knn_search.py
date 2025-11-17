"""
KNN search engine for semantic similarity
"""


import numpy as np
from sklearn.neighbors import NearestNeighbors


from .config import paths




__all__ = ["fit_knn", "load_knn_index", "search_similar"]




# ----------------------
# Build / Save KNN index
# ----------------------


def fit_knn(embeddings: np.ndarray, n_neighbors: int = 5):
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(embeddings)
    return nn




def save_knn_index(nn, path: str = paths.KNN_INDEX_FILE):
    import joblib
    joblib.dump(nn, path)




def load_knn_index(path: str = paths.KNN_INDEX_FILE):
    import joblib
    return joblib.load(path)




# ----------------------
# Similarity Search
# ----------------------


def search_similar(query: str, model, nn, texts, k: int = 5, embed_fn=None):
    if embed_fn is None:
        from .embeddings import get_embeddings as embed_fn

    qvec = embed_fn([query])
    distances, indices = nn.kneighbors(qvec, n_neighbors=k)

    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
        results.append({
            "rank": rank + 1,
            "text": texts[idx],
            "distance": float(dist),
        })

    return results