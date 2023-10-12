import fitz  # import package PyMuPDF
import numpy as np
import pandas as pd

import json
import requests

def get_docs(filename, mem_area):
    docs = fitz.open(filename, stream=mem_area, filetype="pdf")
    return docs

def create_list_of_tables(page):
    list_table_bbox = []
    for table in page.find_tables():
        list_table_bbox.append(table.bbox + (1,))   # Flag: 1: table; 0: text
    return list_table_bbox

def create_list_of_blocks(page):
    list_extract_blocks = []
    for block in page.get_textpage().extractBLOCKS():
        list_extract_blocks.append(block[:4] + (0,))    # Flag: 1: table; 0: text
    return list_extract_blocks

def create_bag_of_bbox(docs):
    list_table_bbox = list(map(lambda page: create_list_of_tables(page), docs))
    list_extract_blocks = list(map(lambda page: create_list_of_blocks(page), docs))

    bag_of_bbox = list(map(lambda x, y: x + y, list_table_bbox, list_extract_blocks))
    return bag_of_bbox

def sort_block(bboxes):
    sorted_list = np.array( sorted(bboxes, key=lambda x: x[1]) )
    max = sorted_list[0][3]
    perfect_list = []
    
    for i, row in enumerate(sorted_list):
        if row[3] < max:
            perfect_list.append(i)
        else:
            max = row[3]
    sorted_list = np.delete(sorted_list, perfect_list, axis = 0)

    return [tuple(row) for row in sorted_list]

'''
input: a page of document,
        bag_of_bbox of above page.
output: content in a page in ordered that stored in a list. Each element is a block of the page.
'''
def generate_text_one_page(page, bag_of_bbox):
    sorted_block_bbox = sort_block(bag_of_bbox)
    content_in_one_page = []
    
    for bbox in sorted_block_bbox:
        if bbox[4] == 1:
            content_in_one_page.append( 'Add table here' )
        elif bbox[4] == 0:
            content_in_one_page.append( page.get_textbox(bbox[:4]) )

    return content_in_one_page

'''
input: a document
output: content of all documents that stored in a list. Each element is a page of the document.
'''
def generate_text_all_pages(docs):
    bag_of_bbox = create_bag_of_bbox(docs)
    return list(map(lambda page, bbox: generate_text_one_page(page, bbox), docs, bag_of_bbox))

def json_to_dataframe(json_table):
    if len(json_table) == 0:
        return []
    
    csv_table = []
    for row in json_table['rows']:
        csv_row = []
        for cell in row['cells']:
            csv_row.append(cell['text'])
        csv_table.append(csv_row)
    
    return pd.DataFrame(csv_table)

def json_to_dataframe_for_one_page(page):
    list_table = []
    for table in page:
        list_table.append(json_to_dataframe(table))
    return list_table

def json_to_dataframe_for_all_pages(docs):
    list_table = []
    for page in docs:
        list_table.append(json_to_dataframe_for_one_page(page))
    return list_table

def concat_tables_to_a_page_content(page, tables):
    table = iter(tables)
    for i, e in enumerate(page):
        try:
            if e == 'Add table here':
                page[i] = next(table).to_json()
            else:
                continue
        except StopIteration:
            break
    return page

def concat_tables_to_all_pages_content(docs, tables):
    return list(map(lambda page, table: concat_tables_to_a_page_content(page, table), docs, tables))

def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]

def transform_list_of_content_to_text(list_content):
    fllaten_list = flatten_comprehension(list_content)

    text = ' '.join(map(str, fllaten_list))                     #concat elements to string.
    text = text.replace('Add table here', '')                   #replace table position that API do not classify and generate value.
    return text
