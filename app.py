import streamlit as st
from RAGAgent import Agent

def main():
    st.title("DocGPT-v2")

    if "agent" not in st.session_state:
        st.session_state.agent = Agent(
            system_prompt=open('core_sys_prompt.txt', 'r').read(),
            verbose=True
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    for history in st.session_state.history:
        with st.chat_message(history["role"]):
            st.markdown(history["text"])

    with st.sidebar:
        st.title("Settings")
        model = st.selectbox("Choose model", options=['qwen2:7b', 'phi3'])
        response_format = st.selectbox("Response format", options=["consice", "detailed"])
        if st.button("Set"):
            st.session_state.agent.model = model
            # st.session_state.agent.
            st.success("Attributes setted.")

        files = st.file_uploader("Upload your files", accept_multiple_files=True, type=["pdf", "txt"])
        process = st.button("Process")
        if process and files:
            with st.spinner('loading your file. This may take a while...'):
                st.session_state.agent.set_document(files)
            st.success("Documents loaded.")

    if prompt := st.chat_input("Enter your message..."):
        st.session_state.history.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            response = st.session_state.agent(prompt, return_addn_queries=False)
            message_placeholder.markdown(response)
        st.session_state.history.append({"role": "assistant", "text": response})

if __name__ == '__main__':
    main()