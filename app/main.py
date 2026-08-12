import time
import logging

from fastapi import FastAPI
from pydantic import BaseModel

from app.retrieval import Retriever
from app.generator import Generator


# ---------------------------------------
# Logging Configuration
# ---------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------
# FastAPI Application
# ---------------------------------------

app = FastAPI(
    title="Cost-Efficient RAG API",
    description="RAG application using ChromaDB",
    version="1.0.0"
)


# ---------------------------------------
# Initialize Components
# ---------------------------------------

retriever = Retriever()
generator = Generator()


# ---------------------------------------
# Request Schema
# ---------------------------------------

class QueryRequest(BaseModel):

    question: str

    top_k: int | None = None


# ---------------------------------------
# Health Endpoint
# ---------------------------------------

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "Cost-Efficient RAG API"
    }


# ---------------------------------------
# Query Endpoint
# ---------------------------------------

@app.post("/query")
def query_rag(request: QueryRequest):

    start_time = time.perf_counter()

    question = request.question.strip()

    if not question:

        return {
            "answer": "Please provide a question.",
            "citations": [],
            "retrieved_chunks": 0
        }

    # -----------------------------------
    # Retrieval
    # -----------------------------------

    if request.top_k is not None:

        local_retriever = Retriever(
            top_k=request.top_k
        )

    else:

        local_retriever = retriever

    retrieved_chunks = local_retriever.retrieve(
        question
    )

    retrieval_latency = (
        time.perf_counter() - start_time
    ) * 1000

    # -----------------------------------
    # Generation
    # -----------------------------------

    generation_result = generator.generate(
        question,
        retrieved_chunks
    )

    total_latency = (
        time.perf_counter() - start_time
    ) * 1000

    # -----------------------------------
    # Logging
    # -----------------------------------

    logger.info(
        "query=%s | chunks=%d | "
        "retrieval_latency_ms=%.2f | "
        "total_latency_ms=%.2f | "
        "input_tokens=%d | "
        "output_tokens=%d | "
        "total_tokens=%d",
        question,
        len(retrieved_chunks),
        retrieval_latency,
        total_latency,
        generation_result["input_tokens"],
        generation_result["output_tokens"],
        generation_result["total_tokens"]
    )

    # -----------------------------------
    # Response
    # -----------------------------------

    return {
        "question": question,

        "answer": generation_result["answer"],

        "citations": generation_result["citations"],

        "retrieved_chunks": len(
            retrieved_chunks
        ),

        "retrieval_latency_ms": round(
            retrieval_latency,
            2
        ),

        "total_latency_ms": round(
            total_latency,
            2
        ),

        "input_tokens": (
            generation_result["input_tokens"]
        ),

        "output_tokens": (
            generation_result["output_tokens"]
        ),

        "total_tokens": (
            generation_result["total_tokens"]
        )
    }