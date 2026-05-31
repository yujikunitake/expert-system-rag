import os
import warnings
import logging
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_embeddings = None
_vectorstore = None

def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings

def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
        if not os.path.exists(db_dir):
            return None
        _vectorstore = Chroma(persist_directory=db_dir, embedding_function=_get_embeddings())
    return _vectorstore

def invalidate_vectorstore_cache():
    """Resets the cached vectorstore instance (call after reindexing)."""
    global _vectorstore
    _vectorstore = None

def search_relevant_documents(query: str, k: int = 3):
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        return [], 999.0

    results = vectorstore.similarity_search_with_score(query, k=k)

    docs = [res[0] for res in results]
    scores = [res[1] for res in results]
    best_score = scores[0] if scores else 999.0

    return docs, best_score
