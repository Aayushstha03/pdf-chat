# langchain document loader class
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader

# langchain document splitter/chunker classes
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# langchain ko built in embeddings class
from langchain_community.embeddings.ollama import OllamaEmbeddings


def load_pdfs_from_directory():
    document_loader = PyPDFDirectoryLoader("docs")
    return document_loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")

documents = load_pdfs_from_directory()
chunks = split_documents(documents)
print(chunks[0])