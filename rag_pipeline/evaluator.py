import os
import json
import faiss
import torch
import gc
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sentence_transformers import SentenceTransformer
from rag_pipeline.model_loader import load_quantized_model
from rag_pipeline.vector_store import load_faiss_index
from rag_pipeline.utils_evaluator import exact_match_score, f1_score

def main():
    print("📊 Starting Evaluation...")

    # Load data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    eval_path = os.path.join(project_root, 'processed_data.json')
    if not os.path.exists(eval_path):
        raise FileNotFoundError(f"Evaluation data not found at {eval_path}")

    with open(eval_path, 'r', encoding='utf-8') as f:
        eval_data = json.load(f)

    eval_data = eval_data[:200]
    print(f"🔍 Loaded {len(eval_data)} evaluation samples")

    # Load components
    print("🧠 Loading embedding model...")
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("📥 Loading FAISS index...")
    index, id_map = load_faiss_index()

    print("⏳ Loading quantized LLM model...")
    tokenizer, model = load_quantized_model()
    device = model.device

    print("⌛ Precomputing query embeddings...")
    questions = [sample.get('question', '') for sample in eval_data]
    query_embeddings = embedder.encode(questions, show_progress_bar=True, convert_to_numpy=True)

    # Metric accumulators
    smoothie = SmoothingFunction().method4
    total_em = total_f1 = total_acc = total_bleu = 0

    results = []
    for idx, sample in enumerate(tqdm(eval_data, desc="Evaluating")):
        question = sample.get('question', '')
        gold_answer = sample.get('answer', '')

        q_emb = query_embeddings[idx:idx+1].astype('float32')
        distances, indices = index.search(q_emb, 3)
        contexts = [eval_data[int(id_map[str(i)])]['context'][:500] for i in indices[0]]

        prompt = (
            "Context:\n" +
            "\n\n".join(contexts) +
            f"\n\nQuestion: {question}\nAnswer:"
        )

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
        acc = 1 if em == 1 else 0
        bleu = sentence_bleu(
            [gold_answer.split()],
            prediction.split(),
            smoothing_function=smoothie
        )

        total_em += em
        total_f1 += f1
        total_acc += acc
        total_bleu += bleu

        results.append({
            'question': question,
            'gold_answer': gold_answer,
            'prediction': prediction,
            'exact_match': em,
            'accuracy': acc,
            'f1_score': f1,
            'bleu_score': bleu
        })

        del inputs, output_ids
        torch.cuda.empty_cache()
        gc.collect()

    # Save detailed results
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, 'eval_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    # Save summary
    summary = {
        'avg_exact_match': total_em / len(eval_data),
        'avg_accuracy': total_acc / len(eval_data),
        'avg_f1_score': total_f1 / len(eval_data),
        'avg_bleu_score': total_bleu / len(eval_data)
    }

    summary_path = os.path.join(results_dir, 'eval_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Evaluation complete. Results saved at: {output_path}")
    print("📊 Evaluation Summary:")
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    nltk.download('punkt')
    main()
