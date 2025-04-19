import argparse
from rag_pipeline import build_index, retriever_generator
from rag_pipeline import evaluator


def main():
    parser = argparse.ArgumentParser(description="🔍 xRAG-LLM Pipeline Controller")
    parser.add_argument('--build', action='store_true', help='📦 Build FAISS index from documents')
    parser.add_argument('--rag', action='store_true', help='🤖 Launch Retrieval + Generation (QA loop)')
    parser.add_argument('--eval', action='store_true', help='📊 Run Evaluation on processed dataset')
    args = parser.parse_args()

    if args.build:
        print("\n🧱 Step 1: Building Index...")
        build_index.main()

    elif args.rag:
        print("\n🚀 Step 2: Starting RAG loop...")
        retriever_generator.main()

    elif args.eval:
        print("\n📊 Step 3: Running Evaluation...")
        evaluator.main()

    else:
        print("\n⚠️  No argument passed. Use:")
        print("   python main.py --build   # to build index")
        print("   python main.py --rag     # to run retriever + generator")
        print("   python main.py --eval    # to run evaluation on dataset")

if __name__ == "__main__":
    main()
    