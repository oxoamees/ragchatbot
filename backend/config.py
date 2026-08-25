"""
config.py
---------
Loads environment variables from .env and exposes them
as simple constants for the rest of the app to import.
"""

import os
from pathlib import Path
import re

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
raw_index_name = os.getenv("PINECONE_INDEX_NAME")
PINECONE_INDEX_NAME = re.sub(r"[^a-z0-9-]", "-", (raw_index_name or "").lower()).strip("-")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY or not PINECONE_INDEX_NAME or not GROQ_API_KEY:
    print("[config.py] WARNING: One or more environment variables are missing. "
          "Check your .env file.")
