import faiss
import numpy as np

def build_faiss_index(embeddings: np.ndarray):
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    
    return index


def search(index, query_embedding: np.ndarray, k = 5):
    query_embedding = query_embedding.astype("float32")
    faiss.normalize_L2(query_embedding)
    D,I = index.search(query_embedding, k)

    return I[0], D[0]