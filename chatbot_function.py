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
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.prompts import AIMessagePromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, PromptTemplate
import streamlit as st

import spacy
nlp = spacy.load("en_core_web_trf", disable=["tagger", "ner", "lemmatizer", "morphologizer", "attribute_ruler"])

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model_name = "gpt-4",
                temperature = 0.3,
                openai_api_key = OPENAI_API_KEY)
#Load docs: not use
def get_docs(filename):
    docs = fitz.open(filename,  
                     filetype = "pdf")
    return docs

#Split docs
def split_docs(docs, chunk_size=3500):
    text_splitter = SpacyTextSplitter(pipeline='en_core_web_trf',
                                      separator='\n\n',
                                      chunk_size=chunk_size)
    all_splits = text_splitter.create_documents([docs])
    return all_splits

#Store to vectorstore
def store_docs_to_vectorstore(all_splits, embedding):
    vectorstore = Chroma.from_documents(documents = all_splits, 
                                        embedding = embedding,
                                        persist_directory = './chroma_db'
                                        )
    return vectorstore

# Generate
def generate_conversation_chain(vectorstore):
    memory = ConversationBufferMemory(memory_key='chat_history', 
                                      return_messages=True)
    system_template = """
    ---
    System: 
    You are the Specialist Business Analytics and you are reading all the information and answer user's question.
    ---
    User: Read me the document
    ---
    Assistant:
    {context}
    ---
    Assitant: You are able to read both text and CSV type in the document.
    ---
    User: {question}
    ---
    Assistant:
    Give an example questions related to user's questions that can query to vector store.
    ---
    User:
    Search from {context} to find the information related to the user's question.
    Thanks to the selectively information from searching above, answer the example question above.
    """

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt}
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