"""
models/schemas.py
------------------
Request/response shapes for the API.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
