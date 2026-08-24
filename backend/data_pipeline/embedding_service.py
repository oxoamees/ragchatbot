"""
data_pipeline/embedding_service.py
-------------------------------------
Responsibility: provide the embedding model used to turn text
into vectors, wrapped the way LangChain expects.

Model: all-MiniLM-L6-v2 (free, runs locally, 384-dimensional vectors)

NOTE: The first time this runs, it downloads the model from
Hugging Face (a few hundred MB) and loads it into memory. That can
take anywhere from a few seconds to a couple of minutes depending
on your internet speed - this is normal, not a bug. After the
first run it's cached locally and loads fast.
"""

print("[embedding_service] Loading embedding model (first run may take a while)...")

from langchain_huggingface import HuggingFaceEmbeddings

# Loaded once and reused everywhere it's imported
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("[embedding_service] Embedding model ready.")


def get_embedding_model():
    """Return the shared embedding model instance."""
    return embedding_model
