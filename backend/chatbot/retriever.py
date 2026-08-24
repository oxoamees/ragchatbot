"""
chatbot/retriever.py
-----------------------
Responsibility: expose a retriever object that, given a question,
returns the top matching chunks from Pinecone.

A "retriever" is just a LangChain wrapper around the vector store's
similarity search, so it can be plugged straight into a chain.
"""

from data_pipeline.vector_store import get_vector_store


def get_retriever(k: int = 3):
    """
    Args:
        k: how many chunks to retrieve per question

    Returns:
        A LangChain retriever object.
    """
    vector_store = get_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})
