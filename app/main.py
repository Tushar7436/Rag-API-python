# app/main.py
from fastapi import FastAPI
from app.schemas import RAGRequest, RAGResponse
from vector_db.search_engine import VectorSearchEngine
from rag.prompt_builder import PromptBuilder
from llm.gemini_client import GeminiClient
from response.formatter import ResponseFormatter
from vector_db.user_history import UserHistoryManager


app = FastAPI()

# ---------- Initialize once (important for performance) ----------
history_manager = UserHistoryManager()
engine = VectorSearchEngine()
llm = GeminiClient()
formatter = ResponseFormatter()
# ---------------------------------------------------------------


@app.post("/history/save")
def save_history(message: str, user_id: str):
    """
    Saves a user message embedding + text in Qdrant.
    """
    history_manager.save_message(user_id, message)
    return {"status": "saved"}


@app.post("/rag", response_model=RAGResponse)
def run_rag(request: RAGRequest):
    """
    End-to-end RAG pipeline:
    1. Retrieve relevant vectors (predefined + user history)
    2. Build final prompt
    3. Generate LLM response
    4. Return formatted answer
    """

    # 1. Retrieve similar stored chunks
    chunks = engine.search_relevant_chunks(
        query=request.message,
        user_id=request.user_id
    )

    # 2. Construct prompt
    prompt = PromptBuilder.build_prompt(
        user_query=request.message,
        context_chunks=chunks
    )

    # 3. Query LLM (Gemini or any other provider)
    ai_output = llm.generate(prompt)

    # 4. Format final output
    output = formatter.format(ai_output)

    return output
