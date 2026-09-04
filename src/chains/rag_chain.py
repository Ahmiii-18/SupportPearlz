from typing import List, Tuple, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from src.config import settings
from src.retrieval.retriever import GatedRetriever
from src.chains.schemas import GroundedResponse
from src.chains.prompts import QA_PROMPT, CONDENSE_QUESTION_PROMPT
from src.utils.logging_setup import logger

class SupportPearlzRAGChain:
    def __init__(self):
        self.retriever = GatedRetriever()
        self.llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            openai_api_key=settings.openai_api_key
        )
        self.structured_llm = self.llm.with_structured_output(GroundedResponse)

    def condense_query(self, question: str, chat_history: List[Any]) -> str:
        if not chat_history:
            return question

        condense_chain = CONDENSE_QUESTION_PROMPT | self.llm | StrOutputParser()
        rewritten = condense_chain.invoke({"chat_history": chat_history, "question": question})
        logger.info(f"Query Condensation: Original='{question}' -> Rewritten='{rewritten}'")
        return rewritten.strip()

    def format_context(self, docs: List[Document]) -> str:
        formatted_blocks = []
        for idx, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "Unknown")
            location = ""
            if "page" in doc.metadata:
                location = f" p.{doc.metadata['page']}"
            elif "row" in doc.metadata:
                location = f" Row {doc.metadata['row']}"
            elif "location" in doc.metadata:
                location = f" {doc.metadata['location']}"

            label = f"[S{idx}] {source}{location}"
            formatted_blocks.append(f"{label}:\n{doc.page_content}")
        return "\n\n".join(formatted_blocks)

    def run(self, question: str, chat_history: List[Any] = None) -> GroundedResponse:
        chat_history = chat_history or []
        
        standalone_query = self.condense_query(question, chat_history)
        retrieved_docs, passed = self.retriever.retrieve(standalone_query)

        if not passed or not retrieved_docs:
            return GroundedResponse(
                answer="I could not find relevant information in the official Pearlz Home Systems documentation regarding your request. Please reach out to customer support at support@pearlzhome.example for assistance.",
                sources=[],
                confidence="none",
                answered=False
            )

        context_str = self.format_context(retrieved_docs)
        
        try:
            formatted_prompt = QA_PROMPT.format_prompt(
                context=context_str,
                chat_history=chat_history,
                question=question
            )
            response: GroundedResponse = self.structured_llm.invoke(formatted_prompt)
            return response
        except Exception as e:
            logger.error(f"Execution or Parsing failure in RAG chain: {str(e)}")
            return GroundedResponse(
                answer="An error occurred while parsing the response from our systems. Please contact support.",
                sources=[],
                confidence="none",
                answered=False
            )