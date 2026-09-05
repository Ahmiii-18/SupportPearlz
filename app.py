import streamlit as st
from src.chains.rag_chain import get_rag_chain

st.set_page_config(
    page_title="SupportPearlz Customer Support",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ SupportPearlz Customer Support Agent")
st.caption("Pearlz Home Systems (Pvt.) Ltd.")

with st.sidebar:
    st.header("Controls")
    if st.button("Reset Chat Memory"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("How can I help you with your Pearlz Home Systems product today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            rag_chain = get_rag_chain()
            response = rag_chain.invoke(prompt)
            
            if isinstance(response, dict):
                answer = response.get("answer", "No response generated.")
                sources = response.get("sources", [])
                confidence = response.get("confidence", "N/A")
            else:
                answer = getattr(response, "answer", str(response))
                sources = getattr(response, "sources", [])
                confidence = getattr(response, "confidence", "N/A")

            st.markdown(answer)

            if sources:
                st.caption(f"**Sources:** {', '.join(sources)} | **Confidence:** {confidence}")

    st.session_state.messages.append({"role": "assistant", "content": answer})