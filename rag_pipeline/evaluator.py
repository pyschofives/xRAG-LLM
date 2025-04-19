import os
import json
import faiss
import torch
import gc
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from rag_pipeline.model_loader import load_quantized_model
from rag_pipeline.vector_store import load_faiss_index
from rag_pipeline.utils_evaluator import exact_match_score, f1_score


def main():
    print("📊 Starting Evaluation...")

    # Load evaluation data from processed_data.json
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    eval_path = os.path.join(project_root, 'processed_data.json')
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation data not found at {eval_path}")

    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    # Use first 200 samples for evaluation
    eval_data = eval_data[:200]
    print(f"🔍 Loaded {len(eval_data)} evaluation samples")

    # Load retrieval components
    print("🧠 Loading embedding model...")
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("📥 Loading FAISS index...")
    index, id_map = load_faiss_index()

    # Load LLM
    print("⏳ Loading quantized LLM model...")
    tokenizer, model = load_quantized_model()
    device = model.device

    # Precompute query embeddings to reduce per-iteration overhead
    questions = [sample.get('question', '') for sample in eval_data]
    print("⌛ Precomputing query embeddings...")
    query_embeddings = embedder.encode(questions, show_progress_bar=True, convert_to_numpy=True)

    results = []
    for idx, sample in enumerate(tqdm(eval_data, desc="Evaluating")):
        question = sample.get('question', '')
        gold_answer = sample.get('answer', '')

        # Retrieve top-k contexts using precomputed embeddings
        q_emb = query_embeddings[idx:idx+1].astype('float32')
        distances, indices = index.search(q_emb, 3)
        # Truncate context to reduce token length
        contexts = [eval_data[int(id_map[str(i)])]['context'][:500] for i in indices[0]]

        # Build prompt with limited context
        prompt = (
            "Context:\n" +
            "\n\n".join(contexts) +
            f"\n\nQuestion: {question}\nAnswer:"
        )

        # Generate answer with inference mode to reduce memory
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                pad_token_id=tokenizer.eos_token_id
            )
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        prediction = prediction.replace(prompt, '').strip()

        # Compute metrics
        em = exact_match_score(prediction, gold_answer)
        f1 = f1_score(prediction, gold_answer)

        results.append({
            'question': question,
            'gold_answer': gold_answer,
            'prediction': prediction,
            'exact_match': em,
            'f1_score': f1
        })

        # Clear memory to avoid fragmentation
        del inputs, output_ids
        torch.cuda.empty_cache()
        gc.collect()

    # Save results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'eval_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    print(f"✅ Evaluation complete. Results saved at: {output_path}")

if __name__ == '__main__':
    main()
