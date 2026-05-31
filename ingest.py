import os
import warnings
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def ingest_data():
    print("Starting data ingestion for the Knowledge Base...")
    
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    db_dir = os.path.join(base_dir, "chroma_db")
    
    if not os.path.exists(data_dir):
        print(f"Error: Folder '{data_dir}' not found.")
        return
        
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        print(f"No text documents found in folder '{data_dir}'.")
        return
        
    print(f"{len(documents)} document(s) loaded. Chunking...")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    
    print(f"Documents split into {len(chunks)} chunks. Generating embeddings...")
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(chunks, embeddings, persist_directory=db_dir)
    
    print(f"Ingestion completed successfully! Vector database saved in '{db_dir}'.")

if __name__ == "__main__":
    ingest_data()
