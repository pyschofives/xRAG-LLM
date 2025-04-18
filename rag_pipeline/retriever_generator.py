import json
import faiss
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from rag_pipeline.model_loader import load_quantized_model

# Constants
INDEX_PATH = "rag_pipeline/faiss_index.index"
ID_MAP_PATH = "rag_pipeline/id_mapping.json"
PROCESSED_DATA_PATH = "processed_data.json"
TOP_K = 3


def retrieve_context(query: str, embedder, index, data, id_map, k=TOP_K):
    """
    Retrieve top-k context passages for a given query.
    """
    query_vec = embedder.encode([query])
    D, I = index.search(np.array(query_vec).astype('float32'), k)

    results = []
    for i in I[0]:
        mapped = id_map.get(str(i))
        idx = mapped['id'] if isinstance(mapped, dict) and 'id' in mapped else mapped
        if idx is not None:
            results.append(data[int(idx)])
    return results


def build_prompt(query: str, contexts):
    """
    Build a prompt by injecting retrieved contexts into the user question.
    """
    context_text = "\n\n".join([f"{i+1}. {c['context']}" for i, c in enumerate(contexts)])
    prompt = (
        "You are an intelligent assistant answering questions based on retrieved context.\n\n"
        f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
    )
    return prompt


def generate_answer(prompt: str, tokenizer, model, max_tokens=256):
    """
    Generate an answer from the LLM given the prompt.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text.replace(prompt, "").strip()


def main():
    # Load index and data
    print("📥 Loading FAISS index and ID map...")
    index = faiss.read_index(INDEX_PATH)
    with open(ID_MAP_PATH, "r") as f:
        id_map = json.load(f)
    with open(PROCESSED_DATA_PATH, "r") as f:
        data = json.load(f)

    # Load embedder and model
    print("🧠 Loading embedding model...")
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("⏳ Loading Quantized LLM model...")
    tokenizer, model = load_quantized_model("models/Qwen2.5-3B-GPTQ")

    # Interactive QA loop
    while True:
        query = input("\n🔍 Enter your question (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Exiting QA loop.")
            break

        contexts = retrieve_context(query, embedder, index, data, id_map)
        print("\n📄 Retrieved Contexts:")
        for i, ctx in enumerate(contexts, 1):
            print(f"{i}. {ctx['context'][:200]}...\n")

        prompt = build_prompt(query, contexts)
        answer = generate_answer(prompt, tokenizer, model)

        print("\n🧠 Answer:\n", answer)


if __name__ == "__main__":
    main()
