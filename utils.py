

def flatten_matrix(two_dimensional_list):
    return [element for sublist in two_dimensional_list for element in sublist]

def deduplicated_documents(docs):
    # Use a set to deduplicate based on content
    deduplicated_documents = list({doc.page_content: doc for doc in docs}.values())
    # Convert the set back to a list to maintain the original order
    return deduplicated_documents