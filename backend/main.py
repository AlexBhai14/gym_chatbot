import sys
import os
import time
from functools import lru_cache

# Allow imports from the backend/ directory regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rag_pipeline import run_rag_pipeline
from retriever import get_retriever         

# ── App initialisation ─────────────────────────────────────────────────────────
app = FastAPI(
    title="🏋️ Gym RAG Chatbot",
    description=(
        "A Retrieval-Augmented Generation chatbot for Gym & Fitness questions. "
        "Uses FAISS + sentence-transformers for retrieval and TinyLlama for generation."
    ),
    version="1.0.0",
)

# Allow all origins for local development (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── ADD 2: Server start hote hi retriever preload karo ─────────────────────────
@app.on_event("startup")
async def preload_models():
    
    print("\n[startup] ⏳ Preloading retriever (embedding model + FAISS index)...")
    get_retriever()
    print("[startup] ✅ Retriever ready! Gym-specific queries will now respond in < 1 sec.\n")

# ── ─────────────────────────────────────────────────────────────────────────────

@lru_cache(maxsize=100)
def cached_rag(query: str) -> dict:
    return run_rag_pipeline(query)

# Serve the simple frontend UI
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")),
    name="static",
)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", summary="Gym Chatbot UI")
def root(request: Request) -> HTMLResponse:
    """Serve the simple chat UI (browser-friendly)."""
    index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    with open(index_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/status", summary="System Status")
def status() -> JSONResponse:
    """Health-check endpoint — confirms the server is running."""
    return JSONResponse(
        content={
            "status":  "✅ Gym RAG Chatbot is running!",
            "version": "1.0.0",
            "model":   "TinyLlama-1.1B-Chat (local)",
            "routes": {
                "GET /":              "Browser chatbot UI",
                "GET /chat?query=":   "Ask the gym chatbot a question",
                "GET /status":        "System status (JSON)",
            },
            "example": "http://127.0.0.1:8000/chat?query=best workout for beginners",
        }
    )


@app.get("/chat", summary="Ask the Gym Chatbot")
def chat(
    query: str = Query(
        ...,
        min_length=3,
        max_length=500,
        description="Your gym or fitness question",
        example="best workout for beginners",
    )
) -> JSONResponse:
   
    start_time = time.perf_counter()

    try:
        result = cached_rag(query.lower().strip())
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Vector store not found. "
                "Please run  `python backend/embed.py`  first "
                "to generate the FAISS index."
            ),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Internal pipeline error: {exc}",
        ) from exc

    elapsed = round(time.perf_counter() - start_time, 2)

    return JSONResponse(
        content={
            "query":   result["query"],
            "answer":  result["answer"],
            "sources": [
                {
                    "rank":     doc["rank"],
                    "score":    round(doc["score"], 4),
                    "question": doc["question"],
                }
                for doc in result["retrieved_docs"]
            ],
            "latency_seconds": elapsed,
        }
    )