import json
import os

def load_triviaqa(path="datasets/triviaqa-unfiltered/unfiltered-web-train.json", max_samples=100):
    """
    Load TriviaQA dataset (unfiltered-web format) and extract (question, context, answer).
    """
    print(f"📂 Loading TriviaQA from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data["Data"][:max_samples]:
        question = item.get("Question", "")
        answer = item.get("Answer", {}).get("Value", "")
        context_list = item.get("SearchResults", [])

        context = ""
        if context_list:
            result = context_list[0]
            context = result.get("Snippet") or result.get("Description") or result.get("Title", "")

        if context.strip() and question and answer:
            samples.append({
                "question": question,
                "context": context,
                "answer": answer
            })

    print(f"✅ Loaded {len(samples)} samples from TriviaQA.")
    return samples

def load_naturalq(path="datasets/simplified-nq-train.jsonl", max_samples=100):
    """
    Load Natural Questions dataset from JSONL format and extract (question, context, answer).
    """
    print(f"📂 Loading NaturalQ from: {path}")
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            try:
                item = json.loads(line)
                question = item.get("question_text", "")
                answer_candidates = item.get("annotations", [{}])[0].get("short_answers", [])
                answer = ""
                if answer_candidates:
                    answer = item.get("document_text", "")[answer_candidates[0]["start_token"]:answer_candidates[0]["end_token"]]
                context = item.get("document_text", "")

                if question and answer and context:
                    samples.append({
                        "question": question,
                        "context": context,
                        "answer": answer
                    })
            except Exception as e:
                continue

    print(f"✅ Loaded {len(samples)} samples from NaturalQ.")
    return samples

if __name__ == "__main__":
    trivia_samples = load_triviaqa()
    nq_samples = load_naturalq()
