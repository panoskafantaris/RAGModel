# Simple in-memory chat manager. Replace with DB for production.
import uuid
import datetime
from typing import Dict, List

CHATS: Dict[str, Dict] = {}

def create_chat(title: str = "New chat") -> str:
    cid = str(uuid.uuid4())
    CHATS[cid] = {
        "id": cid,
        "title": title,
        "messages": [],
        "created": datetime.datetime.utcnow().isoformat(),
        "updated": datetime.datetime.utcnow().isoformat(),
    }
    return cid

def list_chats():
    return [
        {"id": c["id"], "title": c["title"], "last_updated": c["updated"]}
        for c in CHATS.values()
    ]

def append_message(chat_id: str, role: str, content: str):
    entry = {"role": role, "content": content, "ts": datetime.datetime.utcnow().isoformat()}
    CHATS[chat_id]["messages"].append(entry)
    CHATS[chat_id]["updated"] = datetime.datetime.utcnow().isoformat()

def get_history(chat_id: str):
    return CHATS[chat_id]["messages"]