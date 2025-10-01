from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
MODEL_NAME = "ilsp/Llama-Krikri-8B-Instruct"

_tokenizer = None
_model = None
_pipe = None

def init():
    global _tokenizer, _model, _pipe
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if _model is None:
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", torch_dtype="auto")
    if _pipe is None:
        _pipe = pipeline("text-generation", model=_model, tokenizer=_tokenizer, max_new_tokens=512)
    return _pipe

def generate_answer(prompt: str) -> str:
    p = init()
    out = p(prompt, do_sample=True, temperature=0.7)
    text = out[0]["generated_text"]
    # try to strip prompt
    if "Assistant:" in text:
        return text.split("Assistant:")[-1].strip()
    return text