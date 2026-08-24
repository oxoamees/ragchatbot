"""
main.py
-------
Entry point of the app. Exposes ONE endpoint:

  POST /chat  -> takes a question, runs it through the RAG chain
                 (retriever + Groq LLM), and returns the answer.

IMPORTANT: this app only ANSWERS questions, it does not load your
documents into Pinecone. Before using /chat, run ingest.py once
from the terminal:

    python ingest.py

That's a separate one-time step, not part of the running server.
See the note at the bottom of ingest.py for why it's separate.
"""

import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.schemas import ChatRequest, ChatResponse
from chatbot.rag_chain import build_rag_chain

app = FastAPI(title="LangChain RAG Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

print("[main] Starting up - building RAG chain (loading embedding model, connecting to Pinecone and Groq)...")
# Built once at startup and reused for every /chat request
rag_chain = build_rag_chain()
print("[main] Startup complete. Visit http://127.0.0.1:8000/docs")


@app.get("/")
def root():
    return {"message": "LangChain RAG chatbot is running. POST your question to /chat."}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question; get an answer grounded in your indexed documents."""
    try:
        answer = rag_chain.ask(request.question)
        return ChatResponse(answer=answer)

    except Exception as e:
        # Print the FULL traceback to the terminal running uvicorn,
        # so you can see exactly which line failed and why.
        print("\n[main] /chat FAILED - full traceback below:")
        traceback.print_exc()

        # Also return the real error message in the API response itself,
        # instead of a generic "Internal Server Error", so you don't have
        # to go dig through the terminal to find out what went wrong.
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
