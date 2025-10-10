#This loader will load the pdf and will split it into chunks 
#overlapping 1000 chars chunks

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def pdf_load_split(pdf_path = '../data/sample.pdf',chunk_size= 1000, overlap = 150):
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size, chunk_overlap= overlap)
    chunks = splitter.split_documents(docs)

    return chunks
'''
--------
TEST
-------------
chunk = pdf_load_split()
print(len(chunk))
print(chunk[1].page_content[:300])

'''
