"""
data_pipeline/vector_store.py
--------------------------------
Responsibility: connect to Pinecone and provide two things:

1. index_documents()  -> embed + upload chunks into Pinecone (ingest step)
2. get_vector_store()  -> a LangChain-compatible handle used later for search
"""

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from config import PINECONE_API_KEY, PINECONE_INDEX_NAME
from data_pipeline.embedding_service import get_embedding_model
import re

pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = re.sub(r"[^a-z0-9-]", "-", PINECONE_INDEX_NAME.lower()).strip("-")


def _ensure_index_exists():
    """Create the Pinecone index if it doesn't exist yet."""
    existing_indexes = [i["name"] for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  # matches all-MiniLM-L6-v2 output size
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f"[vector_store] Created new Pinecone index: {INDEX_NAME}")


def index_documents(chunks):
    """
    Embed each chunk and upload it into Pinecone.

    Args:
        chunks: list of Document objects (from text_splitter)

    Returns:
        The number of chunks indexed.
    """
    _ensure_index_exists()
    embeddings = get_embedding_model()

    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )

    print(f"[vector_store] Indexed {len(chunks)} chunk(s) into Pinecone.")
    return len(chunks)


def get_vector_store():
    """
    Get a LangChain vector store handle pointing at the existing
    Pinecone index, ready for similarity search.
    """
    _ensure_index_exists()
    embeddings = get_embedding_model()
    return PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
