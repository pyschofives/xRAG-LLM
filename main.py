# main.py (place it at root level or alongside rag_pipeline folder)
import argparse
from rag_pipeline import build_index, retriever_generator

def main():
    parser = argparse.ArgumentParser(description="🔍 xRAG-LLM Pipeline Controller")
    parser.add_argument('--build', action='store_true', help='📦 Build FAISS index from documents')
    parser.add_argument('--rag', action='store_true', help='🤖 Launch Retrieval + Generation (QA loop)')
    args = parser.parse_args()

    if args.build:
        print("\n🧱 Step 1: Building Index...")
        build_index.main()

    elif args.rag:
        print("\n🚀 Step 2: Starting RAG loop...")
        retriever_generator.main()

    else:
        print("\n⚠️  No argument passed. Use:")
        print("   python main.py --build   # to build index")
        print("   python main.py --rag     # to run retriever + generator")

if __name__ == "__main__":
    main()
