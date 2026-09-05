# SupportPearlz AI 🐚🤖

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://supportpearlz-ai.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**SupportPearlz AI** is an intelligent, RAG-based customer support agent designed to deliver accurate, fully grounded answers from official company documentation while eliminating hallucinations.

🌐 **Live Demo:** [supportpearlz-ai.streamlit.app](https://supportpearlz-ai.streamlit.app/)

---

## 🚀 Key Features

- **Gated Retrieval Architecture:** Incorporates strict similarity thresholds to ensure responses are backed by verified context.
- **Multi-Format Ingestion:** Seamlessly processes PDF, DOCX, CSV, and Markdown knowledge base documents.
- **Smart Query Condensation:** Retains conversational context across multi-turn interactions.
- **Interactive UI:** Clean, responsive chat interface built using Streamlit.

---

## 🛠️ Tech Stack

- **Orchestration:** [LangChain](https://www.langchain.com/)
- **Vector Database:** [ChromaDB](https://www.trychroma.com/) (Persistent local store)
- **Embeddings & LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o`)
- **Frontend & UI:** [Streamlit](https://streamlit.io/)
- **Deployment:** Streamlit Community Cloud

---

## 📁 Project Structure

```text
SupportPearlz/
├── data/
│   ├── knowledge_base/     # Source documents (PDF, DOCX, CSV, MD)
│   └── vector_store/       # Persistent ChromaDB vector database
├── src/
│   ├── chains/             # RAG chain logic, prompts, and schemas
│   ├── ingestion/          # Document loaders and index builders
│   ├── retrieval/          # Gated retriever implementation
│   └── utils/              # Logging and helper utilities
├── app.py                  # Main Streamlit application entry point
├── requirements.txt        # Pinned Python dependencies
└── README.md               # Project documentation