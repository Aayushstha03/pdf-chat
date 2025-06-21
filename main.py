from langchain.document_loaders.pdf import PyPDFDirectoryLoader

def load_pdfs_from_directory():
    document_loader = PyPDFDirectoryLoader("data/pdf")
    return document_loader.load()
