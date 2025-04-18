import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_quantized_model(model_path="models/Qwen2.5-3B-GPTQ"):
    """
    Loads the quantized Qwen2.5-3B-GPTQ model and tokenizer.

    Returns:
        tokenizer (AutoTokenizer)
        model (AutoModelForCausalLM)
    """
    print("⏳ Loading Quantized Model from:", model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    model.eval()
    print("✅ Model and tokenizer loaded successfully.")
    return tokenizer, model

# 🔁 Test the loader with a sample QA
if __name__ == "__main__":
    tokenizer, model = load_quantized_model()

    prompt = "What is the capital of Greenland?"
    print(f"\n🧠 Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=50)

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n📤 Response:")
    print(decoded)
