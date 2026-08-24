"""FastAPI entry point for the LangChain RAG chatbot."""

import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import GROQ_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME
from models.schemas import ChatRequest, ChatResponse

app = FastAPI(title="LangChain RAG Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

print("[main] Starting up - building RAG chain (loading embedding model, connecting to Pinecone and Groq)...")
rag_chain = None
startup_error = None

if PINECONE_API_KEY and PINECONE_INDEX_NAME and GROQ_API_KEY:
    try:
        from chatbot.rag_chain import build_rag_chain

        rag_chain = build_rag_chain()
        print("[main] Startup complete. Visit http://127.0.0.1:8000/docs")
    except Exception as error:
        startup_error = f"{type(error).__name__}: {error}"
        print(f"[main] RAG chain unavailable: {startup_error}")
else:
    startup_error = (
        "Missing PINECONE_API_KEY, PINECONE_INDEX_NAME, or GROQ_API_KEY. "
        "Add them to backend/.env and restart the server."
    )
    print(f"[main] RAG chain unavailable: {startup_error}")


@app.get("/")
def root():
    return {
        "message": "LangChain RAG chatbot is running. POST your question to /chat.",
        "ready": rag_chain is not None,
    }


@app.get("/health")
def health():
    return {"status": "ok", "ready": rag_chain is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Ask a question; get an answer grounded in your indexed documents."""
    if rag_chain is None:
        raise HTTPException(status_code=503, detail=startup_error)

    try:
        answer = rag_chain.ask(request.question)
        return ChatResponse(answer=answer)

    except Exception as error:
        print("\n[main] /chat FAILED - full traceback below:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        ) from error
