import json
import requests
import time

API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoiMGVlYzVlMGItODk5MS00NWQyLWEzMTItODY0ZmRiOTQyYzE1IiwidHlwZSI6ImFwaV90b2tlbiJ9.bhobWOHLKfyIl_BCrHA4yeSycb3qKu8unigUrLL2hp8'

headers = {"Authorization": f"Bearer {API_KEY}"}
url = "https://api.edenai.run/v2/ocr/ocr_tables_async"

def post_to_ocr_tables_api(url, files, headers):
    files = {'file': open(files,'rb')}
    data = {"show_original_response": False, 
            "fallback_providers": "", 
            "providers": 'amazon',
            'language':'en'}

    response = requests.post(url, 
                             data = data, 
                             files = files, 
                             headers = headers)

    result = json.loads(response.text)
    return result

def get_job_result_ocr_tables_api(url, public_id, headers):
    url = '/'.join([url, public_id])

    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            res = json.loads(response.text)
            status = res['status']

            if status == 'finished':
                print(f'{status}')
                return res
            else:
                print(f"Status is {status}, waiting...")
                time.sleep(5)
                continue
        else:
            print("Error: Status Code", response.status_code)
            break

def get_list_tables_infor_from_job_result(job_result):
    raw_json = job_result['results']['amazon']['pages']
    return list( map(lambda page: page['tables'], raw_json) )