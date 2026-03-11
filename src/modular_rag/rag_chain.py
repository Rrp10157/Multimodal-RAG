from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from modular_rag.config import SYSTEM_PROMPT
from modular_rag.retrieval import get_relevance_score, resolve_to_parents, semantic_retrieve_and_rerank


def format_docs_with_numbers(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        dtype = doc.metadata.get("type", "text").upper()
        source = doc.metadata.get("source", "")
        score = get_relevance_score(doc)
        score_str = f" score={score:.3f}" if isinstance(score, (int, float)) else ""
        parts.append(f"[{i}] [{dtype}{score_str}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(parts)


def build_rag_chain(compression_retriever, parent_table_map: dict, chat_model):
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])

    def retrieve_and_resolve(query: str) -> str:
        reranked = compression_retriever.invoke(query)
        final = resolve_to_parents(reranked, parent_table_map)
        return format_docs_with_numbers(final)

    return (
        {"context": RunnableLambda(retrieve_and_resolve), "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )


def build_semantic_rag_chain(compression_retriever, parent_table_map: dict, chat_model):
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", "{question}")])

    def retrieve_and_format(query: str) -> str:
        docs = semantic_retrieve_and_rerank(query, compression_retriever, parent_table_map)
        return format_docs_with_numbers(docs)

    return (
        {"context": RunnableLambda(retrieve_and_format), "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | StrOutputParser()
    )
