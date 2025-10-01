from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Load all knowledge docs
loader = DirectoryLoader("data/knowledge", glob="**/*.*", loader_cls=UnstructuredFileLoader)
docs = loader.load()

# Add metadata (store filename)
for d in docs:
    d.metadata["source"] = os.path.basename(d.metadata["source"])

# Split into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create embeddings (multilingual, supports Greek)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Build FAISS vector store
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")

print("✅ Ingestion complete! Files have been indexed.")