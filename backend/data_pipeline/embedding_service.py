"""
data_pipeline/embedding_service.py
-------------------------------------
Responsibility: provide a lightweight 384-dimensional embedding model.

FastEmbed uses ONNX Runtime instead of PyTorch, which keeps the Render
free instance within its memory limit.
"""

print("[embedding_service] Loading embedding model (first run may take a while)...")

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

print("[embedding_service] Embedding model ready.")


def get_embedding_model():
    """Return the shared embedding model instance."""
    return embedding_model
