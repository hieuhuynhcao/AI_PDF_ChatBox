import json

from dotenv import load_dotenv
from transform_raw_docs import *
from chatbot_workflow_function import *
from get_final_extracted_text import *
from html_template import css
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
import pandas as pd

def _main_test():
    load_dotenv()

    # User enters metadata
    if "metadata" not in st.session_state:
        st.session_state.metadata = {}

    # UI
    st.set_page_config(page_title = 'CHAT WITH HIUS', 
                       page_icon = ':sun_with_face:')
    st.write(css, unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header('CHAT WITH HIUS :sun_with_face:')
    user_question = st.text_input('ASK YOUR FUCKING QUESTION ABOUT YOUR DOCS: ')
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.header("Your docs")
        
        # Upload file from local
        pdf_docs = st.file_uploader(
            "Upload your files and click on 'PROCESS'"
            ,accept_multiple_files = True
            ,type = 'pdf'
        )

        if pdf_docs:
            for i, file in enumerate(pdf_docs):
                file_name = file.name
                with st.subheader(f"Metadata for File {i+1}"):
                    if f"metadata_{i}" not in st.session_state:
                        st.session_state[f"metadata_{i}"] = pd.DataFrame(columns=["Customer", "Project", "Document", "Version"])

                    ncol = st.session_state[f"metadata_{i}"].shape[1]

                    with st.form(key=f"add_form_{i}", clear_on_submit=False):
                        st.write(f"Metadata for *{file_name}*")
                        cols = st.columns(ncol)
                        rwdta = []
                        for j in range(ncol):
                            rwdta.append(cols[j].text_input(f"{st.session_state[f'metadata_{i}'].columns[j]}"))

                        if st.form_submit_button("Add"):
                            rw = st.session_state[f"metadata_{i}"].shape[0] + 1
                            st.info("Metadata has been added")
                            st.session_state[f"metadata_{i}"].loc[rw] = rwdta

                    # Update the metadata variable in session state
                    st.session_state.metadata[f"metadata_{i}"] = json.loads(st.session_state[f"metadata_{i}"].transpose().to_json())

        if st.session_state.metadata is not None:
            metadatas = st.session_state.metadata
        else:
            metadatas = 'None'

        list_metadata = []
        for i, _ in enumerate(metadatas):
            metadata = metadatas.get(f'metadata_{i}')
            list_metadata.append(metadata.get('1'))

        if st.button("PROCESS"):
            with st.status("Processing"):
                # get extracted text from pdf file
                list_docs = list(map(lambda doc: get_extracted_text(doc), pdf_docs))

                # Split docs from extrated text
                text_chunks = flatten_comprehension( list(map(lambda doc, metadata: split_docs(doc, metadata), list_docs, list_metadata)) )

                # Add metadata to each page of the docs
                transformered_documents = add_meta_data_to_all_pages(text_chunks)
                st.write(transformered_documents)

                # create vector store
                vectorstore = store_docs_to_vectorstore(transformered_documents, OpenAIEmbeddings())
        
                # create conversation chain
                st.session_state.conversation = generate_conversation_chain(vectorstore)

def main():
    load_dotenv()

    st.set_page_config(page_title = 'CHAT WITH HIUS', 
                       page_icon = ':sun_with_face:')
    st.write(css, unsafe_allow_html = True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header('CHAT WITH HIUS :sun_with_face:')
    user_question = st.text_input('ASK YOUR FUCKING QUESTION ABOUT YOUR DOCS: ')
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your docs")
        # Upload file from local
        pdf_docs = st.file_uploader(
                "Upload your fucking files and click on 'PROCESS'"
                ,accept_multiple_files = True
                ,type = 'pdf')
        
        # User enters metadatas
        if "df" not in st.session_state:
            st.session_state.df = pd.DataFrame(columns = [  "Customer"
                                                            ,"Project"
                                                            ,"Document"
                                                            ,"Version"   ])
        with(st.subheader("Add Metadata")):                                            
            ncol = st.session_state.df.shape[1]
            rw = -1

        # Initialize metadata in session state
        if 'metadata' not in st.session_state:
            st.session_state.metadata = None

        with (st.form(key="add form", clear_on_submit = False)):
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
                pass
                # # get extracted text from pdf file
                # docs = get_extracted_text(pdf_docs)

                # #Split docs from extrated text
                # text_chunks = split_docs(docs, [metadata])

                # #Add metadata to each page of the docs
                # transformered_documents = add_meta_data_to_all_pages(text_chunks)

                # # create vector store
                # vectorstore = store_docs_to_vectorstore(transformered_documents, OpenAIEmbeddings())
        
                # # create conversation chain
                # st.session_state.conversation = generate_conversation_chain(vectorstore)

if __name__ == "__main__":
    # main()
    _main_test()