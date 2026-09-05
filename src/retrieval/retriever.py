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
        return self.get_relevant_documents(query)

    def retrieve(self, query: str):
        """Alias method to support calls expecting .retrieve()"""
        return self.get_relevant_documents(query)

    def get_relevant_documents(self, query: str):
        logger.info(f"Executing retrieval for query: '{query}'")
        
        # Use similarity_search_with_score to get distance metrics (0.0 = exact match)
        results_with_scores = self.vector_store.similarity_search_with_score(
            query, k=self.k
        )
        
        relevant_docs = []
        for doc, distance in results_with_scores:
            # Convert cosine distance to similarity score (1.0 - distance)
            similarity_score = 1.0 - distance
            logger.info(f"Chunk score - Distance: {distance:.4f}, Similarity: {similarity_score:.4f}")
            
            if similarity_score >= self.score_threshold:
                relevant_docs.append(doc)
        
        if not relevant_docs:
            logger.warning(
                f"Relevance Gate Triggered: No retrieved chunks met score threshold ({self.score_threshold})."
            )
            # Return an empty list AND False so unpacking works safely in the chain
            return [], False 
            
        # Return the documents list AND True indicating the gate passed successfully
        return relevant_docs, True

def get_retriever():
    return GatedRetriever()

def retrieve_documents(query: str, score_threshold: float = 0.35):
    retriever = GatedRetriever(score_threshold=score_threshold)
    return retriever.get_relevant_documents(query)