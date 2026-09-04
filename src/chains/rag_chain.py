import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from src.utils.logging_setup import logger

VECTOR_STORE_DIR = "data/vector_store"
COLLECTION_NAME = "supportpearlz_docs"

class GatedRetriever:
    def __init__(self, score_threshold: float = 0.35, k: int = 3):
        self.score_threshold = score_threshold
        self.k = k
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=VECTOR_STORE_DIR,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def invoke(self, query: str):
        docs, _ = self.retrieve(query)
        return docs

    def retrieve(self, query: str) -> tuple:
        """Retrieves documents and returns a tuple: (docs, passed_boolean) to satisfy the RAG chain."""
        logger.info(f"Executing retrieval for query: '{query}'")
        
        results_with_scores = self.vector_store.similarity_search_with_relevance_scores(
            query, k=self.k
        )
        
        relevant_docs = [
            doc for doc, score in results_with_scores if score >= self.score_threshold
        ]
        
        passed = len(relevant_docs) > 0
        
        if not passed:
            logger.warning(
                f"Relevance Gate Triggered: No retrieved chunks met score threshold ({self.score_threshold})."
            )
            
        return relevant_docs, passed

    def get_relevant_documents(self, query: str):
        docs, _ = self.retrieve(query)
        return docs

def get_retriever():
    return GatedRetriever()

def retrieve_documents(query: str, score_threshold: float = 0.35):
    retriever = GatedRetriever(score_threshold=score_threshold)
    docs, _ = retriever.retrieve(query)
    return docs