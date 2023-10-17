import os
import json

from dotenv import load_dotenv
from handle_docs_flow import *
from chatbot_workflow_function import *
from html_template import css
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
import pandas as pd

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
            "Upload your fucking files and click on 'PROCESS'", 
            accept_multiple_files=False, type='pdf')
        
        # User enters metadatas
        if "df" not in st.session_state:
            st.session_state.df = pd.DataFrame(columns=["Customer", 
                                                        "Project", 
                                                        "Document", 
                                                        "Version"])
        with(st.subheader("Add Metadata")):                                            
            ncol = st.session_state.df.shape[1]
            rw = -1

        # Initialize metadata in session state
        if 'metadata' not in st.session_state:
            st.session_state.metadata = None

        with (st.form(key="add form", clear_on_submit = True)):
            cols = st.columns(ncol)
            rwdta = []

            for i in range(ncol):
                rwdta.append(cols[i].text_input(st.session_state.df.columns[i]))
            if st.form_submit_button("Add"):
                rw = st.session_state.df.shape[0] + 1
                st.info("Metadata has been added")
                st.session_state.df.loc[rw] = rwdta

                # Update the metadata variable in session state within the block
                st.session_state.metadata = json.loads(st.session_state.df.transpose().to_json())

        # Now you can use metadata outside of the with st.form block
        if st.session_state.metadata is not None:
            metadata = st.session_state.metadata['1']
        else:
            metadata = 'None'
        st.write(metadata)

        if st.button("PROCESS"):
            with st.status("Processing"):
                # get pdf text
                docs = process_flow(pdf_docs)

                #Split docs
                text_chunks = split_docs(docs, [metadata])
                transformered_documents = add_meta_data_to_all_pages(text_chunks)
                print(transformered_documents)

                # create vector store
                vectorstore = store_docs_to_vectorstore(transformered_documents, OpenAIEmbeddings())
        
                # create conversation chain
                st.session_state.conversation = generate_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()