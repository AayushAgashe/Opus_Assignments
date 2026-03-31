from sentence_transformers import SentenceTransformer
import os

# Force offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

# ABSOLUTE PATH — VERY IMPORTANT
MODEL_PATH = r"D:/PythonTraining/PayFailIntel/local_embedding_model"
model = SentenceTransformer(MODEL_PATH)

def embed(texts):
    return model.encode(texts, convert_to_numpy=True).astype("float32")