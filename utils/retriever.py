import faiss
from utils.embedder import embed_texts, get_embedder
import numpy as np

def build_index(chunks):
    embeddings = embed_texts(chunks)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def retrieve(query, chunks, index, top_k=4):
    model = get_embedder()
    q = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, ids = index.search(q, top_k)
    results = []

    for idx in ids[0]:
        if idx != -1:
            results.append(chunks[idx])

    return results