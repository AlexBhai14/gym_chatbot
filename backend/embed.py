import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH      = os.path.join(BASE_DIR, "data", "gym_knowledge.json")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
INDEX_PATH     = os.path.join(VECTORSTORE_DIR, "gym_faiss.index")
META_PATH      = os.path.join(VECTORSTORE_DIR, "gym_metadata.pkl")

# ── Embedding model ────────────────────────────────────────────────────────────
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_dataset(path: str) -> list[dict]:
    """Load the JSON Q&A dataset from disk."""
    print(f"[embed] Loading dataset from: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig handles Windows BOM
        data = json.load(f)
    print(f"[embed] Loaded {len(data)} entries.")
    return data


def build_documents(data: list[dict]) -> tuple[list[str], list[dict]]:
   
    texts    = []
    metadata = []
    for entry in data:
        combined = f"Q: {entry['question']}\nA: {entry['answer']}"
        texts.append(combined)
        metadata.append({
            "id":       entry["id"],
            "question": entry["question"],
            "answer":   entry["answer"],
        })
    return texts, metadata


def generate_embeddings(texts: list[str], model_name: str) -> np.ndarray:
    """
    Encode texts with SentenceTransformer and L2-normalize the vectors
    so that inner-product search equals cosine similarity.
    """
    print(f"[embed] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"[embed] Generating embeddings for {len(texts)} documents …")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # L2-normalise → inner product == cosine similarity
    faiss.normalize_L2(embeddings)
    print(f"[embed] Embeddings shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS IndexFlatIP (Inner Product) index.
    Because vectors are L2-normalised, IP == cosine similarity.
    """
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)        # cosine similarity via inner product
    index.add(embeddings.astype(np.float32))
    print(f"[embed] FAISS index built  — total vectors: {index.ntotal}")
    return index


def save_artifacts(index: faiss.Index, metadata: list[dict]) -> None:
    """Persist the FAISS index and metadata pickle to the vectorstore folder."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_PATH)
    print(f"[embed] FAISS index saved → {INDEX_PATH}")

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[embed] Metadata saved   → {META_PATH}")


def main() -> None:
    data                 = load_dataset(DATA_PATH)
    texts, metadata      = build_documents(data)
    embeddings           = generate_embeddings(texts, EMBED_MODEL_NAME)
    index                = build_faiss_index(embeddings)
    save_artifacts(index, metadata)
    print("\n✅ Embedding complete! You can now start the FastAPI server.")


if __name__ == "__main__":
    main()