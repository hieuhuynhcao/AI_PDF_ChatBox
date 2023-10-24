from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle5 as pickle

import os
# vectorstore = Chroma(persist_directory='./chroma_db', 
#                      embedding_function=OpenAIEmbeddings())
# vectorstore.persist()
DB_FILE = './faiss_store.pkl'

def load_vector_store():
    # Indexing
    ### Load vector store
    with open("./faiss_store.pkl", "rb") as f:
        vectorstore = pickle.load(f)
    return vectorstore

if os.path.exists(DB_FILE):
    vectorstore = load_vector_store()
else:
    vectorstore = FAISS.from_texts(['Init: This is Coke Knowledge Base'], OpenAIEmbeddings())