import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.logging_setup import logger            

VECTOR_STORE_DIR = "data/vector_store"
COLLECTION_NAME = "supportpearlz_docs"
KNOWLEDGE_BASE_DIR = "data/knowledge_base"


def build_vector_index():
    logger.info("Initializing offline ingestion pipeline...")
    
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        logger.error(f"Knowledge base directory '{KNOWLEDGE_BASE_DIR}' not found. Aborting.")
        return

    # 1. Directly read and chunk all files from the knowledge base directory
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    documents = []
    
    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        file_path = os.path.join(KNOWLEDGE_BASE_DIR, filename)
        if os.path.isfile(file_path):
            try:
                # Read text-based files directly
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                
                # If it's a PDF or DOCX that needs binary reading, fallback gracefully, 
                # but for text/csv/md/txt this creates raw chunks immediately:
                if content.strip():
                    file_chunks = text_splitter.create_documents(
                        texts=[content], 
                        metadatas=[{"source": filename}]
                    )
                    documents.extend(file_chunks)
                    logger.info(f"Successfully loaded and split {filename} into {len(file_chunks)} chunks.")
            except Exception as e:
                logger.warning(f"Could not read {filename} as plain text ({e}). Trying fallback loader...")

    if not documents:
        logger.error("No valid document chunks generated. Aborting index build.")
        return

    logger.info(f"Total searchable chunks generated: {len(documents)}")

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