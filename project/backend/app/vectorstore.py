# Minimal wrapper to load FAISS index and run similarity search
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

EMB_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_DIR = "faiss_index"

_embeddings = None
_db = None

def init_vectorstore():
    global _embeddings, _db
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMB_MODEL)
    if _db is None:
        _db = FAISS.load_local(INDEX_DIR, _embeddings, allow_dangerous_deserialization=True)
    return _db


def retrieve(query: str, k: int = 3):
    db = init_vectorstore()
    docs = db.similarity_search(query, k=k)
    # return tuples (text, metadata)
    return [(d.page_content, d.metadata) for d in docs]