from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model on GPU
EMBED_MODEL = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")

def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Convert a list of texts into numpy embeddings (n, d)
    """
    embeddings = EMBED_MODEL.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    return embeddings


# ---- TEST ----
    print(embed_texts(["I love AI"]).shape)
    