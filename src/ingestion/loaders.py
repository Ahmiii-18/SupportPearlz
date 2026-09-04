import os
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader
)
from src.utils.logging_setup import logger
def load_documents(directory: str):
    logger.info(f"Scanning directory {directory}. Found files/folders.")
    documents = []
    
    if not os.path.exists(directory):
        logger.warning(f"Directory {directory} does not exist.")
        return documents

    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isdir(filepath):
            continue
            
        try:
            if filename.endswith(".csv"):
                loader = CSVLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["doc_type"] = "pricing"
                    doc.metadata["source"] = filename
                documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} records from {filename}")
                
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["doc_type"] = "manual"
                    doc.metadata["source"] = filename
                documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} records from {filename}")
                
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["doc_type"] = "troubleshooting"
                    doc.metadata["source"] = filename
                documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} records from {filename}")
                
            elif filename.endswith(".md"):
                loader = UnstructuredMarkdownLoader(filepath)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["doc_type"] = "warranty"
                    doc.metadata["source"] = filename
                documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} records from {filename}")
                
        except Exception as e:
            logger.error(f"Error loading document {filename}: {e}")
            
    logger.info(f"Total loaded base document passages: {len(documents)}")
    return documents