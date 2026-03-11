import logging
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modular_rag.pipeline import build_pipeline
from modular_rag.retrieval import clean_answer_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def ask(query: str, state: dict):
    answer = state["semantic_rag_chain"].invoke(query)
    return clean_answer_text(str(answer))


def main():
    state = build_pipeline(skip_images=False)
    print("RAG pipeline initialized. Type your question (or 'exit').")
    while True:
        query = input("\n> ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            continue
        print("\n" + ask(query, state))


if __name__ == "__main__":
    main()
