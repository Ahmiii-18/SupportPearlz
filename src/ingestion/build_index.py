import os
import shutil
from dotenv import load_dotenv

# Force load real API key from .env file, overriding terminal session env vars
load_dotenv(override=True)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loaders import load_documents  
from src.utils.logging_setup import logger            

VECTOR_STORE_DIR = "data/vector_store"
COLLECTION_NAME = "supportpearlz_docs"
KNOWLEDGE_BASE_DIR = "data/knowledge_base"


def build_vector_index():
    logger.info("Initializing offline ingestion pipeline...")
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        logger.error(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found. Aborting.")
        return

    raw_documents = load_documents(KNOWLEDGE_BASE_DIR)
    if not raw_documents:
        logger.error(f"No valid documents loaded from {KNOWLEDGE_BASE_DIR}. Aborting.")
        return

    logger.info(f"Successfully loaded {len(raw_documents)} raw document base files.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    
    documents = text_splitter.split_documents(raw_documents)
    logger.info(f"Split raw documents into {len(documents)} searchable chunks.")

    if os.path.exists(VECTOR_STORE_DIR):
        logger.info(f"Removing existing vector store at {VECTOR_STORE_DIR}")
        shutil.rmtree(VECTOR_STORE_DIR)

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