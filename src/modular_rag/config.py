from pathlib import Path


OLLAMA_BASE_URL = "http://localhost:11434"
VISION_MODEL = "llava"
CHAT_MODEL = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

BM25_TOP_K = 15
DENSE_TOP_K = 15
RERANKER_TOP_N = 5

CHROMA_PERSIST_DIR = "./chroma_rag_modular"
MAX_INDEX_CHARS = 1400
INDEX_CHUNK_OVERLAP = 150

SYSTEM_PROMPT = """\
You are a precise financial and economic analyst assistant specialising in Indian economic data from government reports.

INSTRUCTIONS:
1. Answer using ONLY the information in the CONTEXT below.
2. Cite the context block using [1], [2], [3] format for every figure you state.
3. Quote exact numbers as they appear and never round or approximate.
4. For colour-coded table data, include the performance signal when relevant.
5. For box sections, cite them specifically.
6. If the answer is not in the context, say exactly:
   "The provided documents do not contain sufficient information to answer this question."
7. Do NOT use any external knowledge or make assumptions.
8. Keep answers concise and factual.

CONTEXT:
{context}
"""


def default_pdf_sources() -> list[str]:
    project_root = Path(__file__).resolve().parents[2]
    local_pdf = project_root / "Dataset_removed-pages-deleted.pdf"
    return [str(local_pdf.resolve())]
