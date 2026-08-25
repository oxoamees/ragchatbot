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

print("[main] API started. RAG chain will initialize on the first chat request.")
rag_chain = None
startup_error = None


def get_rag_chain():
    global rag_chain, startup_error

    if rag_chain is not None:
        return rag_chain

    if not PINECONE_API_KEY or not PINECONE_INDEX_NAME or not GROQ_API_KEY:
        startup_error = (
            "Missing PINECONE_API_KEY, PINECONE_INDEX_NAME, or GROQ_API_KEY. "
            "Add them to backend/.env and restart the server."
        )
        raise HTTPException(status_code=503, detail=startup_error)

    try:
        from chatbot.rag_chain import build_rag_chain

        rag_chain = build_rag_chain()
        return rag_chain
    except Exception as error:
        startup_error = f"{type(error).__name__}: {error}"
        print(f"[main] RAG chain unavailable: {startup_error}")
        raise HTTPException(status_code=503, detail=startup_error) from error


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
    try:
        answer = get_rag_chain().ask(request.question)
        return ChatResponse(answer=answer)

    except Exception as error:
        print("\n[main] /chat FAILED - full traceback below:")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"{type(error).__name__}: {error}",
        ) from error
