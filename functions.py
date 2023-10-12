from functools import reduce
import json
import operator
import os

from html_template import bot_template, user_template
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from PyPDF2 import PdfReader
import pdf2image
import pytesseract
import streamlit as st
from streamlit_chat import message

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model_name = "gpt-3.5-turbo", 
                temperature = 0,
                openai_api_key = OPENAI_API_KEY)

# Load
def test_load_file_by_selection(list_of_file):

    for file in list_of_file:
        image = pdf2image.convert_from_path('docs_pool/CV_Data.pdf')
        text = ''
        for pagenumber, page in enumerate(image):
            detected_text = pytesseract.image_to_string(page)
            detected_text += detected_text
        text += detected_text
    return text

    # for file in list_of_file:
    #     loader = PdfReader(file)
    #     text = reduce(operator.add, [page.extract_text() + '\n\n' for page in loader.pages])
    # return text

#Split
def split_docs(docs, chunk_size = 1000, chunk_overlap = 200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, 
                                                    chunk_overlap = chunk_overlap)
    all_splits = text_splitter.create_documents([docs])
    return all_splits

#Write splitted docs to file
def write_splited_docs_to_file(file_name, all_splits):
    with open(file_name, "w") as f:
        content = list(map(lambda text: text.dict(), all_splits))
        f.writelines(json.dumps(content))

#Store to vectorstore
def store_docs_to_vectorstore(all_splits, embedding):
    vectorstore = Chroma.from_texts(texts=all_splits, 
                            embedding = embedding,
                            persist_directory = './chroma_db')
    return vectorstore.persist()

#Generate
def generate_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)