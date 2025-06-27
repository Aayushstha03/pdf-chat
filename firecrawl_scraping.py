import argparse
import os
import shutil
from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma

# === CONFIG ===
CHROMA_PATH = "chroma"
DATA_PATH = "docs"

load_dotenv()
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_URLS = [
    "https://www.indianhealthyrecipes.com/butter-chicken/" # put your url(s) here
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("Clearing Database")
        clear_database()

    all_chunks = []

    # 1. Load PDF documents
    print("Loading PDFs...")
    pdf_docs = load_pdf_documents()
    pdf_chunks = split_documents(pdf_docs)
    all_chunks.extend(pdf_chunks)

    # 2. Load Firecrawl documents
    print("Scraping Firecrawl URLs...")
    for url in FIRECRAWL_URLS:
        markdown = fetch_with_firecrawl(url, FIRECRAWL_API_KEY)
        firecrawl_docs = text_to_documents(markdown, source_url=url)
        firecrawl_chunks = calculate_chunk_ids(firecrawl_docs, base=url)
        all_chunks.extend(firecrawl_chunks)

    # 3. Add to Chroma
    print("Adding to Chroma vector DB...")
    add_to_chroma(all_chunks)


# === PDF Processing ===
def load_pdf_documents():
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(documents)



# === Firecrawl Scraping ===
def fetch_with_firecrawl(url: str, api_key: str) -> str:
    app = FirecrawlApp(api_key)
    response = app.scrape_url(url, formats=["markdown"])
    return response.markdown

def text_to_documents(text: str, source_url: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)
    return [Document(page_content=chunk, metadata={"source": source_url}) for chunk in chunks]

# === Chunk ID Generation (PDF + Firecrawl) ===
def calculate_chunk_ids(chunks: list[Document], base: str = "") -> list[Document]:
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source", base)
        page = chunk.metadata.get("page", 0)
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

# === Vector Store Add ===
def add_to_chroma(chunks: list[Document]):
    embeddings = get_embedding_function()
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    existing_items = db.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"Existing docs in DB: {len(existing_ids)}")

    chunks_with_ids = calculate_chunk_ids(chunks)


    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if new_chunks:
        print(f"+ Adding new documents: {len(new_chunks)}")
        new_ids = [doc.metadata["id"] for doc in new_chunks]
        db.add_documents(new_chunks, ids=new_ids)
        db.persist()
    else:
        print("No new documents to add.")

# === Reset Function ===
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

if __name__ == "__main__":
    main()
