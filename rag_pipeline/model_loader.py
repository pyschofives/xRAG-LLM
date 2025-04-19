# rag_pipeline/model_loader.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_quantized_model(model_path="models/Qwen2.5-3B-GPTQ"):
    """
    Loads the GPTQ‑quantized Qwen2.5‑3B model (no BitsAndBytesConfig needed).
    """
    print("⏳ Loading Quantized Model from:", model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    # This will use the GPTQModel/auto-gptq backend automatically
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model.eval()
    print("✅ Model and tokenizer loaded successfully.")
    return tokenizer, model

if __name__ == "__main__":
    tok, m = load_quantized_model()
    # quick smoke test
    inputs = tok("Hello, world!", return_tensors="pt").to(m.device)
    print(tok.decode(m.generate(**inputs)[0], skip_special_tokens=True))
