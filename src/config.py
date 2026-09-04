import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", env="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", env="EMBEDDING_MODEL")
    llm_temperature: float = Field(default=0.0, env="LLM_TEMPERATURE")
    
    vector_store_path: Path = Field(default=Path("./data/vector_store"), env="VECTOR_STORE_PATH")
    collection_name: str = Field(default="pearlz_kb", env="COLLECTION_NAME")
    
    chunk_size: int = Field(default=800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=120, env="CHUNK_OVERLAP")
    retrieval_k: int = Field(default=4, env="RETRIEVAL_K")
    score_threshold: float = Field(default=0.35, env="SCORE_THRESHOLD")
    
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./supportpearlz.log", env="LOG_FILE")

    def get_masked_key(self) -> str:
        if not self.openai_api_key:
            return "NOT_SET"
        return f"{self.openai_api_key[:7]}...{self.openai_api_key[-4:]}"

try:
    settings = Settings()
except Exception as err:
    raise RuntimeError(
        f"Configuration loading failed. Ensure required environment variables are set in .env. Details: {err}"
    )

if __name__ == "__main__":
    print(f"Configuration Loaded Successfully:")
    print(f" - LLM Model: {settings.llm_model}")
    print(f" - Embedding Model: {settings.embedding_model}")
    print(f" - API Key: {settings.get_masked_key()}")
    print(f" - Chunk Size: {settings.chunk_size} (Overlap: {settings.chunk_overlap})")
    print(f" - Vector Path: {settings.vector_store_path}")