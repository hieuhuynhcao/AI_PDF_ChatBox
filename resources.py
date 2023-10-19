from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

import spacy
nlp = spacy.load("en_core_web_trf", 
                 disable=["tagger", "ner", "lemmatizer", "morphologizer", "attribute_ruler"])

vectorstore = Chroma(persist_directory='./chroma_db', 
                     embedding_function=OpenAIEmbeddings())
vectorstore.persist()
