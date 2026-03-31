import re
from langchain_text_splitters import RecursiveCharacterTextSplitter

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40
)


def split_by_sections(text: str):
    """
    Split documentation by logical sections like:
    'Section: Understanding U16'
    """
    text = text.replace("\r", "")
    sections = re.split(r"\n(?=Section:)", text)
    return [s.strip() for s in sections if len(s.strip()) > 50]


def chunk_text(text: str):
    """
    Section‑aware chunking to prevent mixing error codes.
    """
    chunks = []
    for section in split_by_sections(text):
        for c in _splitter.split_text(section):
            if len(c.strip()) > 50:
                chunks.append(c)
    return chunks