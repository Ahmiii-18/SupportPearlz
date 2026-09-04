import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Change 'load_all_documents' to match whatever function exists in your loaders.py
from src.ingestion.loaders import load_documents  # or your actual loader function name
from src.utils.logging_setup import logger             # or adjust import path if needed

VECTOR_STORE_DIR = "data/vector_store"
COLLECTION_NAME = "supportpearlz_docs"


def build_vector_index():
    logger.info("Initializing offline ingestion pipeline...")
    
    # 1. Load documents using your original loader function
    documents = load_documents("data/knowledge_base")
    if not documents:
        logger.error("No valid documents found in data/knowledge_base. Aborting index build.")
        return

    logger.info(f"Loaded document chunks/pages into vector index.")

    # 2. Reset vector store directory
    if os.path.exists(VECTOR_STORE_DIR):
        logger.info(f"Removing existing vector store at {VECTOR_STORE_DIR}")
        shutil.rmtree(VECTOR_STORE_DIR)

    # 3. Create embeddings and initialize Chroma with Cosine space
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    logger.info("Embedding document chunks using text-embedding-3-small...")
    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTOR_STORE_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )

    logger.info(f"Vector database persisted successfully to {VECTOR_STORE_DIR}")
    logger.info("Ingestion pipeline finished successfully. Index updated and persisted to disk.")


if __name__ == "__main__":
    build_vector_index()