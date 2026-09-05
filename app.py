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
        # Display sources if stored in history
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for src in message["sources"]:
                    st.write(f"- {src}")

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
                
                # Extract fields cleanly from GroundedResponse object or dictionary
                if hasattr(response, "answer"):
                    answer = response.answer
                    confidence = getattr(response, "confidence", "N/A")
                    sources = getattr(response, "sources", [])
                elif isinstance(response, dict):
                    answer = response.get("answer", str(response))
                    confidence = response.get("confidence", "N/A")
                    sources = response.get("sources", [])
                else:
                    answer = str(response)
                    confidence = "N/A"
                    sources = []
                
                st.markdown(answer)
                
                if confidence != "N/A" and confidence != "none":
                    st.caption(f"Confidence: {confidence}")
                
                if sources:
                    with st.expander("📚 Sources"):
                        for src in sources:
                            st.write(f"- {src}")
                
                # Save assistant response along with its sources in session state
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": answer,
                    "sources": sources
                })
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