import os
import streamlit as st

# 1. Sync Streamlit Cloud secrets to environment variables
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 2. Apply SQLite patch required by ChromaDB on Linux Cloud containers
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from src.ingestion.build_index import build_vector_index
from src.chains.rag_chain import get_rag_chain

# 3. Force rebuild index on Cloud startup if missing
if not os.path.exists("data/vector_store") or not os.listdir("data/vector_store"):
    with st.spinner("Building vector index from knowledge base..."):
        build_vector_index()

st.set_page_config(page_title="SupportPearlz Customer Support", page_icon="🛡️")
st.title("🛡️ SupportPearlz Customer Support Agent")
st.caption("Pearlz Home Systems (Pvt.) Ltd.")

# Initialize session state for memory
if "chain" not in st.session_state:
    st.session_state.chain = get_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    if st.button("Reset Chat Memory"):
        st.session_state.messages = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
if user_input := st.chat_input("How can I help you with your Pearlz Home Systems product today?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching documentation..."):
            response = st.session_state.chain.invoke(user_input)
            st.write(response.answer)
            if response.sources:
                st.caption(f"Sources: {', '.join(response.sources)} | Confidence: {response.confidence}")

    st.session_state.messages.append({"role": "assistant", "content": response.answer})