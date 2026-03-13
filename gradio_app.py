import sys
from pathlib import Path

import gradio as gr

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modular_rag.pipeline import build_pipeline
from modular_rag.retrieval import clean_answer_text, get_relevance_score, semantic_retrieve_and_rerank

STATE = build_pipeline(skip_images=False)


def _format_context(docs, max_chars: int = 350) -> str:
    lines = []
    for i, doc in enumerate(docs, start=1):
        dtype = str(doc.metadata.get("type", "text")).upper()
        score = get_relevance_score(doc)
        source = str(doc.metadata.get("source", ""))
        snippet = doc.page_content[:max_chars].replace("\n", " ").strip()
        score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "n/a"
        lines.append(f"{i}. [{dtype}] score={score_txt} source={source} :: {snippet}")
    return "\n".join(lines)


def chat_response(message, history):
    query = (message or "").strip()
    if not query:
        return "Please enter a question."
    docs = semantic_retrieve_and_rerank(
        query=query,
        compression_retriever=STATE["compression_retriever"],
        parent_table_map=STATE["parent_table_map"],
        top_n=5,
    )
    answer = STATE["semantic_rag_chain"].invoke(query)
    answer = clean_answer_text(str(answer))
    return f"{answer}\n\nRetrieved context:\n{_format_context(docs)}"


with gr.Blocks(title="Multimodal RAG Chat Interface") as demo:
    gr.Markdown("# Multimodal RAG Chat Interface")
    gr.ChatInterface(fn=chat_response)


if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
