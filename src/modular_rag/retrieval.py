import hashlib
import logging
import re

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from modular_rag.config import CHAT_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)

_semantic_llm = ChatOllama(
    model=CHAT_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.2,
    num_ctx=2048,
)

_hyde_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an economic analyst. Write a short factual answer (2-3 sentences) including likely numbers.",
        ),
        ("human", "{question}"),
    ]
)
_expansion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate exactly 2 alternate phrasings of the question. Output one per line.",
        ),
        ("human", "{question}"),
    ]
)

_hyde_chain = _hyde_prompt | _semantic_llm | StrOutputParser()
_expansion_chain = _expansion_prompt | _semantic_llm | StrOutputParser()


def get_relevance_score(doc: Document) -> float | None:
    for key in ("relevance_score", "score", "rerank_score"):
        raw = doc.metadata.get(key)
        # Flashrank often returns numpy scalar types (e.g. np.float32).
        if hasattr(raw, "item"):
            try:
                raw = raw.item()
            except Exception:
                pass
        if isinstance(raw, (int, float)):
            return float(raw)
        if isinstance(raw, str):
            try:
                return float(raw)
            except ValueError:
                continue
    return None


def resolve_to_parents(docs: list[Document], parent_table_map: dict[str, Document]) -> list[Document]:
    resolved_map: dict[str, Document] = {}
    for doc in docs:
        dtype = doc.metadata.get("type", "")
        parent_id = doc.metadata.get("parent_id")
        child_score = get_relevance_score(doc)

        if dtype == "table_child" and parent_id and parent_id in parent_table_map:
            parent_doc = parent_table_map[parent_id]
            pid = parent_doc.metadata.get("parent_id", parent_id)
            metadata = dict(parent_doc.metadata)
            if child_score is not None:
                metadata["relevance_score"] = child_score
            candidate = Document(page_content=parent_doc.page_content, metadata=metadata)
            existing = resolved_map.get(str(pid))
            if existing is None:
                resolved_map[str(pid)] = candidate
            else:
                existing_score = get_relevance_score(existing)
                if child_score is not None and (existing_score is None or child_score > existing_score):
                    existing.metadata["relevance_score"] = child_score
        else:
            uid = doc.metadata.get("doc_id", id(doc))
            key = str(uid)
            existing = resolved_map.get(key)
            if existing is None:
                resolved_map[key] = doc
            else:
                existing_score = get_relevance_score(existing)
                if child_score is not None and (existing_score is None or child_score > existing_score):
                    existing.metadata["relevance_score"] = child_score
    return list(resolved_map.values())


def semantic_query_layer(query: str) -> list[str]:
    queries = [query]
    try:
        expansion_raw = _expansion_chain.invoke({"question": query})
        expansions = [line.strip() for line in expansion_raw.strip().splitlines() if line.strip()]
        queries.extend(expansions[:2])
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
    try:
        hyde_answer = _hyde_chain.invoke({"question": query})
        if hyde_answer and len(hyde_answer.strip()) > 20:
            queries.append(hyde_answer.strip())
    except Exception as exc:
        logger.warning("HyDE generation failed: %s", exc)
    return queries


def semantic_retrieve_and_rerank(
    query: str,
    compression_retriever,
    parent_table_map: dict[str, Document],
    top_n: int = 5,
) -> list[Document]:
    all_docs: list[Document] = []
    seen_hashes: set[str] = set()

    for variant in semantic_query_layer(query):
        try:
            docs = compression_retriever.invoke(variant)
            for doc in docs:
                digest = hashlib.md5(doc.page_content[:200].encode()).hexdigest()
                if digest not in seen_hashes:
                    seen_hashes.add(digest)
                    all_docs.append(doc)
        except Exception as exc:
            logger.warning("Variant retrieval failed: %s", exc)

    resolved = resolve_to_parents(all_docs, parent_table_map)
    resolved.sort(key=lambda d: get_relevance_score(d) or 0.0, reverse=True)
    return resolved[:top_n]


def clean_answer_text(text: str) -> str:
    text = re.sub(r"(?is)^\s*according to (the )?(provided )?context\s*,?\s*", "", text)
    text = re.sub(r"(?is)^\s*based on (the )?(provided )?context\s*,?\s*", "", text)
    text = re.sub(r"(?i)\baccording to\s*[\[{(]\s*\d+\s*[\]})]\s*,?\s*", "", text)
    text = re.sub(r"\s*\(\s*source\s*:\s*[^)]*\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?i)\s*source\s*:\s*[^.\n]*(?:[.\n]|$)", ". ", text)
    text = re.sub(r"\s*[\[{(]\s*\d+\s*[\]})]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s*\.\s*\.", ".", text)
    return text
