from sentence_transformers import SentenceTransformer
import numpy as np

def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    model = get_embedder()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")
    return embeddings
    