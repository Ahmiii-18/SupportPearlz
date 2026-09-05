from langchain_openai import ChatOpenAI
from src.chains.prompts import RAG_PROMPT
from src.chains.schemas import AgentResponse
from src.retrieval.retriever import get_retriever
from src.utils.logging_setup import logger


class SupportPearlzChain:
    def __init__(self):
        self.retriever = get_retriever()
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        self.structured_llm = self.llm.with_structured_output(AgentResponse)
        self.prompt = RAG_PROMPT

    def invoke(self, question: str) -> AgentResponse:
        # 1. Retrieve relevant passages via GatedRetriever
        docs, gate_passed = self.retriever.get_relevant_documents(question)

        # 2. Build context based on retrieval status
        if not gate_passed or not docs:
            logger.info(f"Relevance gate blocked retrieval or no docs found for query: '{question}'")
            context_text = "No relevant internal documentation found."
            retrieved_sources = []
        else:
            context_text = "\n\n".join([doc.page_content for doc in docs])
            retrieved_sources = list(
                set([doc.metadata.get("source", "Unknown") for doc in docs if doc.metadata])
            )

        # 3. Generate structured Pydantic response (handles greetings or fallback via system prompt)
        formatted_prompt = self.prompt.format_messages(context=context_text, question=question)
        response = self.structured_llm.invoke(formatted_prompt)

        # 4. Attach source metadata if unpopulated by LLM
        if not response.sources and retrieved_sources:
            response.sources = retrieved_sources

        return response


def get_rag_chain():
    return SupportPearlzChain()