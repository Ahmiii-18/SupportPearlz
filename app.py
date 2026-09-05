import os
import streamlit as st

# Sync Streamlit Cloud secrets to environment variables
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

from src.ingestion.build_index import build_vector_index
from src.chains.rag_chain import get_rag_chain

# Force clean vector index initialization on application startup
@st.cache_resource(show_spinner="Initializing knowledge base index...")
def init_vector_store():
    build_vector_index()
    return True

init_vector_store()

st.set_page_config(page_title="SupportPearlz Customer Support", page_icon="🛡️")
st.title("🛡️ SupportPearlz Customer Support Agent")
st.caption("Pearlz Home Systems (Pvt.) Ltd.")

if "chain" not in st.session_state:
    st.session_state.chain = get_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Controls and Suggested Questions
clicked_question = None
with st.sidebar:
    st.header("Controls")
    if st.button("Reset Chat Memory", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.header("💡 Suggested Questions")
    
    suggested_queries = [
        "What is covered under the Pearlz warranty policy?",
        "What are the available plans and prices?",
        "How do I install my Pearlz Home System?",
        "What steps should I follow if my device stops responding?",
        "Who are you and what do you do?"
    ]
    
    for q in suggested_queries:
        if st.button(q, key=f"btn_{q}", use_container_width=True):
            clicked_question = q

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Capture input from text prompt or sidebar button click
chat_input_value = st.chat_input("How can I help you with your Pearlz Home Systems product today?")
user_input = clicked_question or chat_input_value

if user_input:
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
    if clicked_question:
        st.rerun()