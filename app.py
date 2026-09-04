import streamlit as st
from src.chains.rag_chain import SupportPearlzRAGChain

# Page Configuration
st.set_page_config(
    page_title="SupportPearlz Customer Support Agent",
    page_icon="💬",
    layout="centered"
)

# App Header
st.markdown("## 🛡️ SupportPearlz Customer Support Agent")
st.markdown("*Pearlz Home Systems (Pvt.) Ltd.*")
st.markdown("---")

# Initialize RAG Chain in Session State
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = SupportPearlzRAGChain()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Handling
if prompt := st.chat_input("How can I help you with your Pearlz Home Systems product today?"):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Bot Response
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and generating response..."):
            try:
                response = st.session_state.rag_chain.run(prompt)
                
                # Handle response structure (dict or string)
                if isinstance(response, dict):
                    answer = response.get("answer", str(response))
                    confidence = response.get("confidence", "N/A")
                else:
                    answer = str(response)
                    confidence = "N/A"
                
                st.markdown(answer)
                if confidence != "N/A":
                    st.caption(f"Confidence: {confidence}")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                error_msg = f"An error occurred while processing your request: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Sidebar Controls
with st.sidebar:
    st.markdown("### Controls")
    if st.button("Reset Chat Memory"):
        st.session_state.messages = []
        if hasattr(st.session_state.rag_chain, "reset_memory"):
            st.session_state.rag_chain.reset_memory()
        st.success("Chat history cleared!")
        st.rerun()