from langchain_community.document_loaders import PyPDFLoader
from fastapi import UploadFile

async def extract_text_from_pdf(file: UploadFile):
    file_content = await file.read()
    
    with open("temp_pdf.pdf", "wb") as f:
        f.write(file_content)

    loader = PyPDFLoader("temp_pdf.pdf")
    pages = loader.load_and_split()
    
    text = ""
    for page in pages:
        text += page.page_content
    return text
