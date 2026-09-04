
# SupportPearlz Customer Support Agent

## 🚀 Live Demo

You can try the live application here: [SupportPearlz AI Customer Support Agent](https://supportpearlz-ai.streamlit.app/)

SupportPearlz is an offline-first Retrieval-Augmented Generation (RAG) assistant built for **Pearlz Home Systems (Pvt.) Ltd.** It processes technical product manuals, troubleshooting guides, warranty policies, and pricing data sheets to accurately resolve customer service inquiries while incorporating a confidence scoring and relevance gating mechanism.

---

## Project Structure

```text
SupportPearlz/
│
├── data/
│   ├── knowledge_base/        # Source documents (PDF, DOCX, CSV, MD)
│   └── vector_store/          # Persisted ChromaDB vector database
│
├── evaluation/                # Evaluation suite and test questions
│
├── src/
│   ├── chains/                # RAG chain implementation & prompt logic
│   ├── ingestion/             # Document loaders and vector index builder
│   ├── retrieval/             # Gated retriever with cosine similarity search
│   ├── utils/                 # Logging setup and configuration
│   ├── app_cli.py             # CLI Interface entry point
│   └── config.py              # Environment and system settings
│
├── app.py                     # Streamlit Web UI interface
├── requirements.txt           # Project Python dependencies
└── README.md                  # Project Documentation