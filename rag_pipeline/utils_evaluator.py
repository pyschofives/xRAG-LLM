import re
import string
from collections import Counter


def normalize_answer(text: str) -> str:
    """
    Lowercase, remove punctuation, articles and extra whitespace.
    """
    def lower(x): return x.lower()
    def remove_articles(x): return re.sub(r"\b(a|an|the)\b", " ", x)
    def remove_punctuation(x): return x.translate(str.maketrans('', '', string.punctuation))
    def white_space_fix(x): return ' '.join(x.split())

    return white_space_fix(remove_articles(remove_punctuation(lower(text))))


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """
    Computes Exact Match (EM): 1 if normalized texts match exactly, else 0.
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Computes F1 score at token-level between prediction and ground truth.
    """
    pred_tokens = normalize_answer(prediction).split()
    true_tokens = normalize_answer(ground_truth).split()
    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())

    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return float(pred_tokens == true_tokens)
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    return 2 * precision * recall / (precision + recall)
