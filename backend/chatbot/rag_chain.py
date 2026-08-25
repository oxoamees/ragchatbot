"""
chatbot/rag_chain.py
------------------------
Responsibility: wire the retriever and the LLM together into a
single RAG step: question in -> retrieved context -> answer out.

Written as plain, explicit Python (no fancy chain-builder helpers)
so it stays correct across LangChain versions and is easy to read
for beginners: retrieve -> build prompt -> ask the LLM -> return text.
"""

from chatbot.retriever import get_retriever
from chatbot.llm_service import get_llm

PROMPT_TEMPLATE = """Answer the question clearly and helpfully.
Use the document context below when it is available and relevant.
If the context is empty, answer general knowledge questions using your built-in knowledge.
Do not invent facts about the user's private documents when they are not in the context.

Context:
{context}

Question:
{question}

Answer:"""


class RagChain:
    """Small wrapper so main.py can just call rag_chain.ask(question)."""

    def __init__(self):
        self.retriever = get_retriever()
        self.llm = get_llm()

    def ask(self, question: str) -> str:
        # Step 1: retrieve the most relevant chunks for this question
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs) if docs else "No document context is indexed yet."

        # Step 2: build the final prompt
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        # Step 3: ask the LLM
        response = self.llm.invoke(prompt)

        # ChatGroq (like other chat models) returns a message object;
        # the actual text is in .content
        return response.content


def build_rag_chain() -> RagChain:
    """
    Build and return the RAG pipeline.

    Usage:
        rag_chain = build_rag_chain()
        answer = rag_chain.ask("your question")
    """
    return RagChain()
