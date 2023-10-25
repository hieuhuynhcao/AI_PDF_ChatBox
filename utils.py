from typing import List
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")
class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)

def flatten_matrix(two_dimensional_list):
    return [element for sublist in two_dimensional_list for element in sublist]

def deduplicated_documents(docs):
    # Use a set to deduplicate based on content
    deduplicated_documents = list({doc.page_content: doc for doc in docs}.values())
    # Convert the set back to a list to maintain the original order
    return deduplicated_documents