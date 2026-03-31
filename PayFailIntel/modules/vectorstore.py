import faiss
import pandas as pd
import os

# Global absolute output directory for FAISS storage
DEFAULT_OUTPUT_DIR = os.path.abspath(
    "D:/PythonTraining/PayFailIntel/outputs"
)

def build_index(emb):
    """Build a FAISS L2 index from embeddings."""
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index


def save(index, df, path=None):
    """
    Save FAISS index + metadata.
    If no path is given, saves to DEFAULT_OUTPUT_DIR.
    """
    # Resolve path
    if path is None:
        path = DEFAULT_OUTPUT_DIR

    path = os.path.abspath(path)
    os.makedirs(path, exist_ok=True)

    index_path = os.path.join(path, "faiss_index.bin")
    meta_path  = os.path.join(path, "faiss_metadata.csv")

    faiss.write_index(index, index_path)
    df.to_csv(meta_path, index=False)

    print(f"[vectorstore] Saved index → {index_path}")
    print(f"[vectorstore] Saved metadata → {meta_path}")


def load(path=None):
    """
    Load FAISS index + metadata.
    If no path is given, loads from DEFAULT_OUTPUT_DIR.
    """
    # Resolve path
    if path is None:
        path = DEFAULT_OUTPUT_DIR

    path = os.path.abspath(path)

    index_path = os.path.join(path, "faiss_index.bin")
    meta_path  = os.path.join(path, "faiss_metadata.csv")

    # Check files exist
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"[vectorstore] FAISS index not found:\n{index_path}"
        )

    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"[vectorstore] Metadata not found:\n{meta_path}"
        )

    index = faiss.read_index(index_path)
    meta = pd.read_csv(meta_path)

    print(f"[vectorstore] Loaded index ← {index_path}")
    print(f"[vectorstore] Loaded metadata ← {meta_path}")

    return index, meta