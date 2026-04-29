import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
INDEX_PATH     = os.path.join(VECTORSTORE_DIR, "gym_faiss.index")
META_PATH      = os.path.join(VECTORSTORE_DIR, "gym_metadata.pkl")

# ── Config ─────────────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K            = 3          # number of documents to retrieve


class GymRetriever:
    

    def __init__(self) -> None:
        print("[retriever] Loading embedding model …")
        self.model = SentenceTransformer(EMBED_MODEL_NAME)

        print("[retriever] Loading FAISS index …")
        if not os.path.exists(INDEX_PATH):
            raise FileNotFoundError(
                f"FAISS index not found at {INDEX_PATH}. "
                "Please run  `python backend/embed.py`  first."
            )
        self.index = faiss.read_index(INDEX_PATH)

        print("[retriever] Loading metadata …")
        with open(META_PATH, "rb") as f:
            self.metadata: list[dict] = pickle.load(f)

        print(f"[retriever] Ready — index has {self.index.ntotal} vectors.")

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
    
        # 1. Encode the user query into an embedding vector
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
        ).astype(np.float32)

        # 2. L2-normalise so inner product = cosine similarity
        faiss.normalize_L2(query_vector)

        # 3. Search the FAISS index for top_k nearest neighbours
        distances, indices = self.index.search(query_vector, top_k)

        # 4. Collect and return results
        results = []
        for rank, (score, idx) in enumerate(
            zip(distances[0], indices[0]), start=1
        ):
            if idx == -1:          # FAISS returns -1 when fewer results exist
                continue
            meta = self.metadata[idx]
            results.append({
                "rank":     rank,
                "score":    float(score),
                "question": meta["question"],
                "answer":   meta["answer"],
            })

        print(
            f"[retriever] Query: '{query}' → "
            f"retrieved {len(results)} docs "
            f"(scores: {[round(r['score'], 3) for r in results]})"
        )
        return results


# ── Module-level singleton (lazy-loaded on first import) ──────────────────────
_retriever_instance: GymRetriever | None = None


def get_retriever() -> GymRetriever:
    """Return the shared GymRetriever instance (created on first call)."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = GymRetriever()
    return _retriever_instance
