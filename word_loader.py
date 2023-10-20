from docx import Document
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
from docx.table import _Cell, Table, _Row
from docx.text.paragraph import Paragraph

# Initialize variables to track heading levels
heading_levels = [0, 0, 0]  # Initialize levels for Heading 1, 2, and 3

# Define a function to update heading numbers
def update_heading_numbers(heading_level):
    heading_levels[heading_level - 1] += 1  # Increment the current level
    # Reset lower-level headings
    for i in range(heading_level, 3):
        heading_levels[i] = 0
    # Generate the heading number
    heading_number = '.'.join(str(level) for level in heading_levels[:heading_level])
    return heading_number

def iter_block_items(parent):
    """
    Generate a reference to each paragraph and table child within *parent*,
    in document order. Each returned value is an instance of either Table or
    Paragraph. *parent* would most commonly be a reference to a main
    Document object, but also works for a _Cell object, which itself can
    contain paragraphs and tables.
    """
    if isinstance(parent, _Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    elif isinstance(parent, _Row):
        parent_elm = parent._tr
    else:
        raise ValueError("something's not right")
    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)

def transform_text(filename):
    list_content = []
    document = Document(filename)
    
    for block in iter_block_items(document):
        heading_number = None
        if isinstance(block, Paragraph):
            if block.style.name.split()[0] == 'Heading':
                heading_level = int(block.style.name.split()[-1])
                heading_number = update_heading_numbers(heading_level)
                if heading_number:
                    list_content.append(f"{heading_number} {block.text}")
                else:
                   list_content.append(block.text)

        elif isinstance(block, Table):
            for i, row in enumerate(block.rows):
                text = (cell.text for cell in row.cells)
                if i == 0:
                    keys = tuple(text)
                    continue
                row_data = dict(zip(keys, text))
                formatted_string = "\n".join([f"{key}: {value}" for key, value in row_data.items()])
                list_content.append(f'---\n{formatted_string}')

    return '\n'.join(list_content)