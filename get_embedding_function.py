# langchain ko built in embeddings class
from langchain_ollama import OllamaEmbeddings

# langchain class to handle chrome db creation and stuffs
from langchain_chroma import Chroma

def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")
