from chatbot_workflow_function import *
from transform_raw_docs import *
from handle_ocr_table_api import *

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiNmJmZDE4NWUtYzk0Zi00MjBhLWE4NTUtODg1MDFjY2MxZTIxIiwidHlwZSI6ImFwaV90b2tlbiJ9.5x5zxatYCWo8EJfTkp6rzDYXAFOpDr-nUganZdbUIa4'

headers = {"Authorization": f"Bearer {API_KEY}"}
url = "https://api.edenai.run/v2/ocr/ocr_tables_async"

# Extract only 1 pdf file!
def get_extracted_text(pdf_docs):
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
    public_id = post_result['public_id']

    job_result = get_job_result_ocr_tables_api(url, public_id, headers)

    list_tables_infor = get_list_tables_infor_from_job_result(job_result)
    pretty_tables = json_to_dataframe_for_all_pages(list_tables_infor)

    # Transform to final docs
    list_of_all_content = concat_tables_to_all_pages_content(pretty_raw_docs, pretty_tables)
    final_docs = transform_list_of_content_to_text(list_of_all_content)

    return final_docs