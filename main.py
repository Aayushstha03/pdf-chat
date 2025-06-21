# langchain document loader class
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

# langchain document splitter/chunker classes
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def load_pdfs_from_directory():
    document_loader = PyPDFDirectoryLoader("docs")
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

#multiple chunks can have the same page and hence be the same, need update page id

documents = load_pdfs_from_directory()
chunks = split_documents(documents)
print(chunks[0])