import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Paths
DATA_PATH = "processed_data.json"  # Path to the processed data file
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'faiss_index.index')
ID_MAP_PATH = os.path.join(os.path.dirname(__file__), 'id_mapping.json')


def main():
    # Load processed data
    print("📥 Loading processed data...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Load embedding model
    print("🧠 Loading embedding model: all-MiniLM-L6-v2")
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Embed documents
    print("🔍 Embedding documents...")
    texts = [entry['context'] for entry in data]
    embeddings = embedder.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Build FAISS index
    print("🔧 Building FAISS index...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_PATH)
    print(f"✅ FAISS index saved at: {INDEX_PATH}")

    # Save ID map {faiss_idx: original_data_idx}
    id_map = {str(i): i for i in range(len(data))}
    with open(ID_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(id_map, f, indent=2)
    print(f"✅ ID map saved at: {ID_MAP_PATH}")


if __name__ == '__main__':
    main()
