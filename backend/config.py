"""
config.py
---------
Loads environment variables from .env and exposes them
as simple constants for the rest of the app to import.
"""

import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME or not GROQ_API_KEY:
    print("[config.py] WARNING: One or more environment variables are missing. "
          "Check your .env file.")
