import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from process_flow import *
from html_template import css
import streamlit as st
from langchain.vectorstores import Chroma

def main():
    load_dotenv()

    st.set_page_config(page_title = 'CHAT WITH HIUS', 
                       page_icon = ':sun_with_face:')
    st.write(css, unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header('CHAT WITH HIUS :sun_with_face:')
    user_question = st.text_input('ASK YOUR FUCKING QUESTION ABOUT YOUR DOCS: ')
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your docs")
        pdf_docs = st.file_uploader(
            "Upload your fucking files and click on 'PROCESS'", accept_multiple_files=False, type='pdf')
        
        if st.button("PROCESS"):
            with st.status("Processing"):
                # get pdf text
                docs = process_flow(pdf_docs)
                st.write(docs)

                #Split docs
                text_chunks = split_docs(docs)

                # create vector store
                vectorstore = store_docs_to_vectorstore(text_chunks, OpenAIEmbeddings())
        
                # create conversation chain
                st.session_state.conversation = generate_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()