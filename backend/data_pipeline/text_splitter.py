"""
data_pipeline/text_splitter.py
---------------------------------
Responsibility: break large documents into small overlapping chunks.

Why: embeddings work best on short passages, and Pinecone returns
whole chunks as "context" — chunks that are too big waste tokens,
chunks that are too small lose meaning. ~500 characters with a
little overlap is a solid beginner default.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Split LangChain Document objects into smaller chunks.

    Args:
        documents: list of Document objects (from document_loader)
        chunk_size: max characters per chunk
        chunk_overlap: characters shared between consecutive chunks

    Returns:
        A list of smaller Document objects (the chunks).
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    print(f"[text_splitter] Split into {len(chunks)} chunk(s).")
    return chunks
