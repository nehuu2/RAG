import os
from dotenv import load_dotenv

# Load variables from .env
load_dotenv()

# -----------------------------
# Chunking Configuration
# -----------------------------

CHUNK_SIZE = int(
    os.getenv("CHUNK_SIZE", "1000")
)

CHUNK_OVERLAP = int(
    os.getenv("CHUNK_OVERLAP", "200")
)

# -----------------------------
# Retrieval Configuration
# -----------------------------

TOP_K = int(
    os.getenv("TOP_K", "5")
)

SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", "0.20")
)

# -----------------------------
# Embedding Configuration
# -----------------------------

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# LLM Configuration
# -----------------------------

LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "https://api.openai.com/v1"
)

LLM_API_KEY = os.getenv(
    "LLM_API_KEY",
    ""
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "gpt-4o-mini"
)

# -----------------------------
# ChromaDB Configuration
# -----------------------------

CHROMA_PATH = "chroma_db"

COLLECTION_NAME = "rag_documents"