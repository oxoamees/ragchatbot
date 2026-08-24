"""
data_pipeline/document_loader.py
----------------------------------
Responsibility: read raw files from data/documents/ and turn them
into LangChain Document objects (text + metadata).

Supports .pdf and .txt files. Drop your files into data/documents/
before running the ingest step.
"""

import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader

DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "documents")


def load_documents():
    """
    Load every supported file inside data/documents/.

    Returns:
        A list of LangChain Document objects.
    """
    documents = []

    if not os.path.exists(DOCUMENTS_DIR):
        print(f"[document_loader] No documents folder found at {DOCUMENTS_DIR}")
        return documents

    for filename in os.listdir(DOCUMENTS_DIR):
        filepath = os.path.join(DOCUMENTS_DIR, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(filepath)
            documents.extend(loader.load())

        elif filename.lower().endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            documents.extend(loader.load())

        else:
            print(f"[document_loader] Skipping unsupported file: {filename}")

    print(f"[document_loader] Loaded {len(documents)} raw document(s).")
    return documents
