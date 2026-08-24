"""
chatbot/llm_service.py
-------------------------
Responsibility: provide the LLM (hosted on Groq) that generates
the final answer, wrapped the way LangChain expects.

NOTE ON MODEL NAME: Groq retires old model IDs over time (e.g.
llama-3.1-8b-instant and llama-3.3-70b-versatile were deprecated).
If you ever get a "model_not_found" / 404 error here, check the
current list at https://console.groq.com/docs/models and swap the
GROQ_MODEL value below.
"""

from langchain_groq import ChatGroq
from config import GROQ_API_KEY

# openai/gpt-oss-20b: smaller/faster, good default for a simple chatbot
# openai/gpt-oss-120b: bigger/smarter, swap to this if you need better answers
GROQ_MODEL = "openai/gpt-oss-20b"


def get_llm():
    """Return a LangChain-compatible Groq chat model."""
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        temperature=0.2,
    )
