import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from src.config import settings
from src.utils.logging_setup import logger


class VectorStoreManager:
    """Manages initialization, creation, updating, and querying of the Chroma vector store."""

    def __init__(self) -> None:
        self.persist_path = Path(settings.vector_store_path)
        self.collection_name = settings.collection_name
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )

    def get_vector_store(self) -> Chroma:
        """Loads and returns an existing Chroma vector store instance."""
        if not self.persist_path.exists():
            raise FileNotFoundError(
                f"Vector store directory '{self.persist_path}' does not exist. "
                "Run your index building script first."
            )
        return Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(self.persist_path),
        )

    def create_index(
        self, documents: List[Document], force_rebuild: bool = True
    ) -> Chroma:
        """Creates or overwrites the vector store index with the provided documents."""
        if force_rebuild and self.persist_path.exists():
            logger.info(f"Removing existing vector store at {self.persist_path}")
            shutil.rmtree(self.persist_path)

        self.persist_path.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Embedding {len(documents)} document chunks using {settings.embedding_model}..."
        )

        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=str(self.persist_path),
        )
        logger.info(f"Vector database persisted successfully to {self.persist_path}")
        return vector_store

    def add_documents(self, documents: List[Document]) -> List[str]:
        """Appends new documents to an existing vector store index without rebuilding."""
        store = self.get_vector_store()
        ids = store.add_documents(documents)
        logger.info(f"Added {len(documents)} document chunks to vector store.")
        return ids

    def similarity_search(
        self, query: str, k: int = 4, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Document]:
        """Performs standard similarity search against the vector store."""
        store = self.get_vector_store()
        return store.similarity_search(query=query, k=k, filter=metadata_filter)

    def similarity_search_with_scores(
        self, query: str, k: int = 4, metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """Performs similarity search returning document chunks with relevance scores."""
        store = self.get_vector_store()
        return store.similarity_search_with_relevance_scores(
            query=query, k=k, filter=metadata_filter
        )

    def as_retriever(
        self, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None
    ):
        """Exposes the vector store as a standard LangChain retriever interface."""
        store = self.get_vector_store()
        search_kwargs = search_kwargs or {"k": 4}
        return store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )

    def clear_index(self) -> None:
        """Deletes the persisted vector store directory."""
        if self.persist_path.exists():
            shutil.rmtree(self.persist_path)
            logger.info(f"Cleared vector store directory at {self.persist_path}")
        else:
            logger.warning(f"No vector store directory found at {self.persist_path} to clear.")