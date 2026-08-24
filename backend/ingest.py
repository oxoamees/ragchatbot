"""
ingest.py
---------
Run this file directly from the terminal, ONCE, before you use the
chatbot (and again any time you add new files to data/documents/):

    python ingest.py

WHY THIS IS A SEPARATE SCRIPT AND NOT AN API ENDPOINT:
Ingestion (reading files, splitting them, embedding them, uploading
to Pinecone) is a one-time data-prep job, not something a chat user
should ever trigger by hitting an endpoint. Keeping it as its own
script means:
  - main.py only has to do one job: answer questions
  - you can't accidentally re-index everything from a stray API call
  - it's obvious when indexing is happening (you ran it yourself)

WHAT IT DOES:
  1. Loads every .pdf / .txt file from data/documents/
  2. Splits them into small chunks
  3. Embeds each chunk and uploads it to your Pinecone index
"""

from data_pipeline.document_loader import load_documents
from data_pipeline.text_splitter import split_documents
from data_pipeline.vector_store import index_documents


def main():
    print("Step 1/3: Loading documents from data/documents/ ...")
    documents = load_documents()

    if not documents:
        print("No documents found. Add a .pdf or .txt file to data/documents/ and try again.")
        return

    print("Step 2/3: Splitting documents into chunks ...")
    chunks = split_documents(documents)

    print("Step 3/3: Embedding and uploading chunks to Pinecone ...")
    count = index_documents(chunks)

    print(f"Done. {count} chunk(s) indexed. You can now run: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
