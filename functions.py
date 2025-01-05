from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Read the PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# splitting PDF text into chunks
def get_text_chunks(text):
    # using langchain splitter to chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks
