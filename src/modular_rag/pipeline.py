import logging

from langchain_core.documents import Document

from modular_rag.config import INDEX_CHUNK_OVERLAP, MAX_INDEX_CHARS, default_pdf_sources
from modular_rag.document_processing import (
    convert_documents,
    extract_and_merge_boxes,
    extract_image_descriptions,
    extract_table_chunks,
    extract_text_chunks,
    stitch_multipage_tables,
)
from modular_rag.index_manager import build_compression_retriever, build_hybrid_retriever
from modular_rag.models import load_models
from modular_rag.rag_chain import build_rag_chain, build_semantic_rag_chain

logger = logging.getLogger(__name__)


def _split_long_docs(
    docs: list[Document],
    max_chars: int = MAX_INDEX_CHARS,
    overlap: int = INDEX_CHUNK_OVERLAP,
) -> list[Document]:
    split_docs: list[Document] = []
    for doc in docs:
        text = doc.page_content or ""
        if len(text) <= max_chars:
            split_docs.append(doc)
            continue

        step = max(1, max_chars - overlap)
        for i, start in enumerate(range(0, len(text), step)):
            chunk = text[start : start + max_chars]
            if not chunk.strip():
                continue
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = i
            metadata["chunk_total"] = (len(text) + step - 1) // step
            split_docs.append(Document(page_content=chunk, metadata=metadata))

    return split_docs


def build_pipeline(pdf_sources: list[str] | None = None, skip_images: bool = False):
    sources = pdf_sources or default_pdf_sources()
    embeddings_model, chat_model = load_models()

    conversions = convert_documents(sources)
    if not conversions:
        raise RuntimeError("No documents converted. Verify PDF paths.")

    text_docs = extract_text_chunks(conversions)
    text_docs, box_docs = extract_and_merge_boxes(text_docs)
    table_parents, table_children = extract_table_chunks(
        conversions,
        start_id=len(text_docs) + len(box_docs),
    )
    table_parents, table_children = stitch_multipage_tables(table_parents, table_children)
    image_docs = [] if skip_images else extract_image_descriptions(
        conversions,
        start_id=(len(text_docs) + len(box_docs) + len(table_parents) + len(table_children)),
    )

    parent_table_map: dict[str, Document] = {doc.metadata["parent_id"]: doc for doc in table_parents}
    indexable_docs = text_docs + box_docs + table_children + image_docs
    indexable_docs = _split_long_docs(indexable_docs)
    ensemble_retriever, _ = build_hybrid_retriever(indexable_docs, embeddings_model)
    compression_retriever = build_compression_retriever(ensemble_retriever)

    rag_chain = build_rag_chain(compression_retriever, parent_table_map, chat_model)
    semantic_rag_chain = build_semantic_rag_chain(compression_retriever, parent_table_map, chat_model)

    logger.info(
        "Pipeline ready | text=%s box=%s tables=%s table_rows=%s images=%s",
        len(text_docs),
        len(box_docs),
        len(table_parents),
        len(table_children),
        len(image_docs),
    )
    return {
        "rag_chain": rag_chain,
        "semantic_rag_chain": semantic_rag_chain,
        "compression_retriever": compression_retriever,
        "parent_table_map": parent_table_map,
    }
