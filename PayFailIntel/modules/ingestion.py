import pandas as pd
import docx
import pypdf

def read_pdf(file):
    reader = pypdf.PdfReader(file)
    return "\n".join([page.extract_text() for page in reader.pages])

def read_txt(file):
    return file.read().decode("utf-8")

def read_docx(file):
    d = docx.Document(file)
    return "\n".join([p.text for p in d.paragraphs])

def read_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def extract_text_from_file(f):
    name = f.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(f)
    if name.endswith(".txt"):
        return read_txt(f)
    if name.endswith(".docx"):
        return read_docx(f)
    if name.endswith(".csv"):
        return read_csv(f)
    return ""