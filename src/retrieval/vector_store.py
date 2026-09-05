from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

VECTOR_STORE_DIR = "data/vector_store"
COLLECTION_NAME = "supportpearlz_docs"

def get_vector_store() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=VECTOR_STORE_DIR,
        collection_metadata={"hnsw:space": "cosine"}
    )