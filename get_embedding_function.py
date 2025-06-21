# langchain ko built in embeddings class
from langchain_community.embeddings.ollama import OllamaEmbeddings

# langchain class to handle chrome db creation and stuffs
from langchain_community.vectorstores.chroma import Chroma

def get_embeddings():
    return OllamaEmbeddings(model="nomic-embed-text")
