from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .models import NewChatRequest, MessageRequest, ChatSummary, UploadResponse
from .chat_manager import create_chat, list_chats, append_message, get_history
from .vectorstore import retrieve
from .ingestion import save_upload
from .llm import generate_answer

app = FastAPI()

# Enable CORS for Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Chat endpoints (multi-chat)
# ------------------------------

@app.post("/chats", response_model=str)
def create_new_chat(req: NewChatRequest):
    """Create a new chat session and return its ID."""
    return create_chat(title=req.title or "New chat")

@app.get("/chats")
def get_chats():
    """List all chats (id + summary)."""
    return list_chats()

@app.post("/chats/{chat_id}/message")
def post_message(chat_id: str, msg: MessageRequest):
    """Send a message inside a specific chat and get assistant reply."""
    # Append user message
    append_message(chat_id, msg.role, msg.content)
    
    # Retrieve relevant context from vectorstore
    docs = retrieve(msg.content, k=3)
    context_text = "\n\n".join([f"[Source: {m.get('source','?')}]\n{text}" for text, m in docs])

    # Build prompt
    system_instruction = "You are IT assistant. Answer in Greek. Use provided context when possible."
    history = get_history(chat_id)
    prompt = system_instruction + "\n\n"

    for h in history:
        prompt += f"{h['role']}: {h['content']}\n"

    prompt += f"\nCONTEXT:\n{context_text}\n\nAssistant:"

    # Call LLM
    answer = generate_answer(prompt)

    # Append assistant reply to history
    append_message(chat_id, "assistant", answer)

    return {"answer": answer, "sources": [m for _, m in docs]}

@app.get("/chats/{chat_id}/history")
def get_chat_history(chat_id: str):
    """Return the full history of a specific chat."""
    return get_history(chat_id)

# ------------------------------
# Upload endpoint
# ------------------------------

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a new file to knowledge base."""
    path = await save_upload(file)
    return {"filename": file.filename, "status": "saved"}

# ------------------------------
# Simple stateless chat endpoint
# ------------------------------

@app.post("/chat")
def simple_chat(msg: MessageRequest):
    """Quick one-shot Q&A without multi-chat history."""
    docs = retrieve(msg.content, k=3)
    context_text = "\n\n".join([f"[Source: {m.get('source','?')}]\n{text}" for text, m in docs])

    system_instruction = "You are IT assistant. Answer in Greek. Use provided context when possible."
    prompt = f"{system_instruction}\n\nUser: {msg.content}\n\nContext:\n{context_text}\n\nAssistant:"

    answer = generate_answer(prompt)
    return {"answer": answer, "sources": [m for _, m in docs]}

# ------------------------------
# Health check
# ------------------------------

@app.get("/ping")
def ping():
    return {"ok": True}
