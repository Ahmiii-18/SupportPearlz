from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.logging_setup import logger

def chunk_documents(
    documents: List[Document], 
    chunk_size: int = 300, 
    chunk_overlap: int = 50
) -> List[Document]:
    logger.info(f"Chunking {len(documents)} raw document passages...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Generated {len(chunks)} document chunks.")
    return chunks