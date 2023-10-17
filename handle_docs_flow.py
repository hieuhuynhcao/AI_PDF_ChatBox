import os

from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from chatbot_workflow_function import *
from handle_raw_docs import *
from handle_ocr_table_api import *
from html_template import css
import streamlit as st
from langchain.vectorstores import Chroma

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiZWM1Zjc2YmMtYTc5Yy00MzM1LWFmMDYtMzk4YjlhYTUxNWYyIiwidHlwZSI6ImFwaV90b2tlbiJ9.JJ2S_2_UP4AN-Vqm6bbafltPnZBjJ-4wpi9_eN9HCPg'

headers = {"Authorization": f"Bearer {API_KEY}"}
url = "https://api.edenai.run/v2/ocr/ocr_tables_async"

def process_flow(pdf_docs):
    file_contents = pdf_docs.read()
    with fitz.open(stream = file_contents, filetype = "pdf") as raw_docs:
        pretty_raw_docs = generate_text_all_pages(raw_docs)
    
    # Table OCR API
    files = {"file": (pdf_docs.name, file_contents)}
    data = {"show_original_response": False, 
            "fallback_providers": "", 
            "providers": 'amazon',
            'language':'en'}
    
    response = requests.post(url, 
                            data = data,
                            files = files, 
                            headers = headers)
    post_result = json.loads(response.text)

    #Handle tables in docs
    # post_result = post_to_ocr_tables_api(url, pdf_docs, headers)
    public_id = post_result['public_id']

    job_result = get_job_result_ocr_tables_api(url, public_id, headers)

    list_tables_infor = get_list_tables_infor_from_job_result(job_result)
    pretty_tables = json_to_dataframe_for_all_pages(list_tables_infor)

    # Transform to final docs
    list_of_all_content = concat_tables_to_all_pages_content(pretty_raw_docs, pretty_tables)
    final_docs = transform_list_of_content_to_text(list_of_all_content)

    return final_docs