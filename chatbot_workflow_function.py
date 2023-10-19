from html_template import bot_template, user_template

import fitz  # import package PyMuPDF
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import SpacyTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import streamlit as st

import spacy
nlp = spacy.load("en_core_web_trf", 
                 disable=["tagger", "ner", "lemmatizer", "morphologizer", "attribute_ruler"])

# OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
# llm = ChatOpenAI(model_name = "gpt-4",
#                 temperature = 0.2,
#                 openai_api_key = OPENAI_API_KEY)
#Load docs
def get_docs(filename, mem_area):
    docs = fitz.open(filename, stream=mem_area, filetype="pdf")
    return docs

#Split docs
def split_docs(docs, metadata, chunk_size=4000):
    text_splitter = SpacyTextSplitter(pipeline='en_core_web_trf',
                                      separator='\n\n',
                                      chunk_size=chunk_size
                                      ,chunk_overlap=500)
    all_splits = text_splitter.create_documents([docs]
                                                ,metadatas = [metadata])
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
    system_template = """You are a Splecialist Business Analytics. End every answer with the pretty metadata. Use the following pieces of context to answer the users question. 
        If you cannot find the answer from the pieces of context, just say that you don't know, don't try to make up an answer.
        ---
        {context}
        ---
        User: {question}
        ---
        Assitant:
        Paraphrase the user's question

        """
    # qa_prompt = ChatPromptTemplate.from_template(system_template)

    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    conversation_chain = ConversationalRetrievalChain.from_llm(
            llm = ChatOpenAI(temperature = 0.4
                            ,model = "gpt-4"
                            ,max_tokens = 1000)
            ,retriever = vectorstore.as_retriever()
            # condense_question_llm = ChatOpenAI(temperature = 0.4, 
            #                                    model = 'gpt-4'),
            #condense_question_prompt = qa_prompt,
            ,combine_docs_chain_kwargs = {'prompt': qa_prompt}
            ,memory = memory
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