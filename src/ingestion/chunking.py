from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import settings
from src.utils.logging_setup import logger

class DocumentChunker:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            keep_separator=True
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks: List[Document] = []
        for doc in documents:
            if doc.metadata.get("doc_type") == "pricing":
                chunks.append(doc)
            else:
                split_chunks = self.text_splitter.split_documents([doc])
                for idx, chunk in enumerate(split_chunks):
                    chunk.metadata["chunk_id"] = f"{chunk.metadata.get('source', 'doc')}_c{idx}"
                    chunks.append(chunk)

        logger.info(f"Transformed {len(documents)} document objects into {len(chunks)} chunk vectors.")
        return chunks