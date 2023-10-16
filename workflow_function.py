from functools import reduce
import json
import operator
import os

import fitz  # import package PyMuPDF
from html_template import bot_template, user_template
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.text_splitter import SpacyTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate
import streamlit as st

import spacy
nlp = spacy.load("en_core_web_trf", 
                 disable=["tagger", "ner", "lemmatizer", "morphologizer", "attribute_ruler"])

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model_name = "gpt-4",
                temperature = 0.2,
                openai_api_key = OPENAI_API_KEY)
#Load docs
def get_docs(filename, mem_area):
    docs = fitz.open(filename, stream=mem_area, filetype="pdf")
    return docs

#Split docs
def split_docs(docs, metadata, chunk_size=3500):
    text_splitter = SpacyTextSplitter(pipeline='en_core_web_trf',
                                      separator='\n\n',
                                      chunk_size=chunk_size)
    all_splits = text_splitter.create_documents([docs]
                                                ,metadatas=metadata)
    return all_splits

#Store to vectorstore
def store_docs_to_vectorstore(all_splits, embedding):
    vectorstore = Chroma.from_texts(texts = all_splits, 
                                        embedding = embedding,
                                        persist_directory = './chroma_db'
                                        )
    return vectorstore

#Generate
def generate_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True)
    system_template = """
    ---
    System:
    You are the Splecialist Business Analytics. You are reading all the information and answer user's question.
    ---
    User: Read me the document and the whole metadata.
    ---
    Assitant: 
    Given the following conversation and a follow up question, rephrase the follow up question to be 3 standalone questions.
    ---
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:

    """


    qa_prompt = ChatPromptTemplate.from_template(system_template)

    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm = ChatOpenAI(temperature = 0, 
                            model = "gpt-4",
                            max_tokens = 1000),
            retriever = vectorstore.as_retriever(),
            condense_question_llm = ChatOpenAI(temperature = 0, 
                                               model = 'gpt-4'),
            condense_question_prompt = qa_prompt,
            memory = memory
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