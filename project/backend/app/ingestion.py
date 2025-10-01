# Lightweight ingestion endpoint to add uploaded files to the `data/knowledge` folder
# and (re)run your ingestion script (index/build). For POC we'll save files and call
# your ingestion script or re-index function.
import os
from fastapi import UploadFile

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "knowledge")

async def save_upload(file: UploadFile) -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, file.filename)
    with open(path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    # For POC: call your ingestion pipeline (synchronously or enqueue)
    # from ingestion_script import run_ingestion
    # run_ingestion() # or reindex changed files
    return path