import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# Constants
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PROCESSED_DATA_PATH = "processed_data.json"
INDEX_SAVE_PATH = "rag_pipeline/faiss_index.index"
MAPPING_SAVE_PATH = "rag_pipeline/id_mapping.json"


def load_processed_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def embed_documents(docs, model):
    embeddings = []
    for doc in tqdm(docs, desc="🔍 Embedding documents"):
        text = doc["context"]
        emb = model.encode(text, show_progress_bar=False, normalize_embeddings=True)
        embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)


def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index


def save_index(index, path):
    faiss.write_index(index, path)


def save_id_mapping(data, path):
    mapping = {i: {"question": x["question"], "context": x["context"], "answer": x["answer"]} for i, x in enumerate(data)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)


if __name__ == "__main__":
    print("📥 Loading processed data...")
    documents = load_processed_data(PROCESSED_DATA_PATH)

    print("🧠 Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("🔧 Embedding and building FAISS index...")
    embeddings = embed_documents(documents, model)
    index = build_faiss_index(embeddings)

    print("💾 Saving index and ID mappings...")
    save_index(index, INDEX_SAVE_PATH)
    save_id_mapping(documents, MAPPING_SAVE_PATH)

    print(f"✅ FAISS index built and saved at {INDEX_SAVE_PATH}")