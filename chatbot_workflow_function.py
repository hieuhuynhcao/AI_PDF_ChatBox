import fitz  # import package PyMuPDF
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA, LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import SpacyTextSplitter, RecursiveCharacterTextSplitter
import pickle5 as pickle

from resources import vectorstore
from utils import flatten_matrix, deduplicated_documents, LineListOutputParser
from langchain.retrievers.multi_query import MultiQueryRetriever

#Load pdf docs: not use
def _get_docs(filename, mem_area):
    docs = fitz.open(filename, stream=mem_area, filetype="pdf")
    return docs

#Split docs
def split_docs_with_recursive(docs,metadata, chunk_size=2000):
    custom_text_splitter = RecursiveCharacterTextSplitter(
        # Set custom chunk size
        chunk_size = chunk_size,
        chunk_overlap  = 0,
        length_function = len,
        separators=['---']
    )
    all_splits = custom_text_splitter.create_documents([docs],
                                                       metadatas=[metadata])
    return all_splits

# Not use
def _split_docs(docs, metadata, chunk_size=4000):
    text_splitter = SpacyTextSplitter(pipeline='en_core_web_trf'
                                      ,separator='\n\n'
                                      ,chunk_size=chunk_size
                                      ,chunk_overlap=0)
    all_splits = text_splitter.create_documents([docs]
                                                ,metadatas = [metadata])
    return all_splits

#Store to vectorstore
def store_docs_to_vectorstore(all_splits):
    vectorstore.add_texts(texts = all_splits)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

#Generate
def create_exam_questions(question):
    CREATE_EXAMPLE_QUESTIONS_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""
        Act as a Prompt Enhancer AI that takes user-input prompts and transforms them into more engaging, detailed, and thought-provoking questions. Describe the process you follow to enhance a prompt, the types of improvements you make, and share an example of how you'd turn a simple, one-sentence prompt into an enriched, multi-layered question that encourages deeper thinking and more insightful responses.
        Your task is to generate three different versions of the given user question to retrieve relevant documents from a vector 
        database. By generating multiple perspectives on the user question, your goal is to help
        the user overcome some of the limitations of the distance-based similarity search. 
        Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )
    # Chain
    llm_chain = LLMChain(
                        llm = ChatOpenAI(temperature = 0
                                    ,model = "gpt-3.5-turbo"
                                    ,max_tokens = 1000)
                         ,prompt = CREATE_EXAMPLE_QUESTIONS_PROMPT
                         ,output_parser = LineListOutputParser())

    # Run
    retriever = MultiQueryRetriever(
        retriever = vectorstore.as_retriever(), 
        llm_chain = llm_chain, 
        parser_key = "lines"
    )  # "lines" is the key (attribute name) of the parsed output

    # Results
    unique_docs = retriever.get_relevant_documents(query=question)
    return unique_docs

def create_paraphrase_questions(user_question):
    qa_chain = RetrievalQA.from_chain_type(
            llm = ChatOpenAI(model_name="gpt-3.5-turbo",
                            temperature=0,
                            max_tokens=800),
            retriever=vectorstore.as_retriever()
            )
    result = qa_chain({"query": user_question})
    
    context_doc = flatten_matrix( list(map(lambda q: vectorstore.similarity_search(q), result['result'].split('\n'))) )
    return deduplicated_documents(context_doc)

def generate_conversation_chain(question, vectorstore):
    QUERY_PROMPT = PromptTemplate(
        input_variables=['question', 'context'],
        template="""
        System: You are a Specialist Data Analytics.
        ---
        User:
        {question}
        ---
        Assitant: Use only the context to answer the user's question. Just answer the main point.
        {context}
        
        ---
        Assitant:

        
    """
    )
    conversation_chain = RetrievalQA.from_chain_type(
            llm = ChatOpenAI(temperature = 0
                            ,model = "gpt-3.5-turbo"
                            ,max_tokens = 1000)
            ,retriever = vectorstore.as_retriever()
            ,chain_type_kwargs={ "prompt": QUERY_PROMPT }
            ,verbose=True
    )
    #print(len(create_paraphrase_questions(question)))
    result = conversation_chain({'query': question,
                                 'context': "".join(doc.page_content for doc in create_exam_questions(question))})
    return result

# def handle_userinput(user_question):
#     response = st.session_state.conversation({"query": user_question})
#     st.write(response)
    # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)
    #         print(response)
    #     else:
    #         st.write(bot_template.replace(
    #             "{{MSG}}", message.content), unsafe_allow_html=True)