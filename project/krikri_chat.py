from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

MODEL_NAME = "ilsp/Llama-Krikri-8B-Instruct"

# Load model
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

# Load FAISS DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Load instructions into system prompt
instruction_text = ""
for fname in os.listdir("data/instructions"):
    with open(os.path.join("data/instructions", fname), "r", encoding="utf-8") as f:
        instruction_text += f.read() + "\n"

def retrieve_context(query):
    docs = db.similarity_search(query, k=3)
    return "\n\n".join([f"[Source: {d.metadata.get('source','?')}]\n{d.page_content}" for d in docs])

def ask(question):
    # Retrieve relevant knowledge
    context = retrieve_context(question)
    
    # Build prompt with instructions + retrieved context
    prompt = f"""
SYSTEM:
{instruction_text}

CONTEXT from knowledge files:
{context}

USER: {question}
ASSISTANT:
"""
    result = pipe(prompt, do_sample=True, temperature=0.7)[0]["generated_text"]
    answer = result.split("ASSISTANT:")[-1].strip()
    return answer

if __name__ == "__main__":
    print("Greek IT Assistant (with files + instructions)")
    print("Type 'exit' to quit.\n")
    while True:
        q = input("Εσύ: ")
        if q.lower().strip() == "exit":
            break
        reply = ask(q)
        print(f"Βοηθός: {reply}\n")
