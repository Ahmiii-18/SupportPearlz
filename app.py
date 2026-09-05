import os
import streamlit as st

# Sync Streamlit Cloud secrets to environment variables
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from src.ingestion.build_index import build_vector_index
from src.chains.rag_chain import get_rag_chain

# Automatically build vector database on startup if missing
if not os.path.exists("data/vector_store") or not os.listdir("data/vector_store"):
    with st.spinner("Initializing knowledge base index..."):
        build_vector_index()

st.set_page_config(page_title="SupportPearlz Customer Support", page_icon="🛡️")
st.title("🛡️ SupportPearlz Customer Support Agent")
st.caption("Pearlz Home Systems (Pvt.) Ltd.")

if "chain" not in st.session_state:
    st.session_state.chain = get_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Controls")
    if st.button("Reset Chat Memory"):
        st.session_state.messages = []
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

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