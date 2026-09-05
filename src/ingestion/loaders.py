import os
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    TextLoader
)
from src.utils.logging_setup import logger

def load_documents(directory_path: str) -> List[Document]:
    logger.info(f"Scanning directory {directory_path}. Found files/folders.")
    documents = []

    if not os.path.exists(directory_path):
        logger.error(f"Directory {directory_path} does not exist.")
        return documents

    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if not os.path.isfile(file_path):
            continue

        ext = os.path.splitext(file)[1].lower()
        loaded_docs = []

        try:
            if ext == ".csv":
                loader = CSVLoader(file_path=file_path)
                loaded_docs = loader.load()
            elif ext == ".pdf":
                loader = PyPDFLoader(file_path)
                loaded_docs = loader.load()
            elif ext in [".docx", ".doc"]:
                loader = Docx2txtLoader(file_path)
                loaded_docs = loader.load()
            elif ext in [".md", ".markdown"]:
                loader = UnstructuredMarkdownLoader(file_path)
                loaded_docs = loader.load()
            elif ext == ".txt":
                loader = TextLoader(file_path, encoding="utf-8")
                loaded_docs = loader.load()

            if loaded_docs:
                for doc in loaded_docs:
                    doc.metadata["source"] = file
                documents.extend(loaded_docs)
                logger.info(f"Successfully loaded {len(loaded_docs)} records from {file}")
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

    logger.info(f"Total loaded base document passages: {len(documents)}")
    return documents