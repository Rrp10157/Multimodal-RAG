import logging

from langchain_chroma import Chroma
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langchain_classic.retrievers import ContextualCompressionRetriever

from modular_rag.config import BM25_TOP_K, CHROMA_PERSIST_DIR, DENSE_TOP_K, RERANKER_TOP_N

logger = logging.getLogger(__name__)


def build_hybrid_retriever(all_indexable_docs, embeddings_model):
    from langchain_community.retrievers import BM25Retriever
    from langchain_classic.retrievers import EnsembleRetriever

    bm25_retriever = BM25Retriever.from_documents(all_indexable_docs, k=BM25_TOP_K)
    dense_vectorstore = Chroma.from_documents(
        documents=all_indexable_docs,
        embedding=embeddings_model,
        collection_name="hybrid_dense_final",
        persist_directory=CHROMA_PERSIST_DIR + "_dense",
    )
    dense_retriever = dense_vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": DENSE_TOP_K},
    )
    ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever],
        weights=[0.4, 0.6],
    )
    logger.info("Hybrid retriever ready")
    return ensemble, dense_vectorstore


def build_compression_retriever(base_retriever) -> ContextualCompressionRetriever:
    reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=RERANKER_TOP_N)
    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
