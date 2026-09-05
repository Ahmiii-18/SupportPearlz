import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

KNOWLEDGE_BASE_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "data", "vector_store")
COLLECTION_NAME = "supportpearlz_docs"

SCORE_THRESHOLD = 0.35
RETRIEVAL_K = 3
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50