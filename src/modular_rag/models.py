import logging

from langchain_ollama import ChatOllama, OllamaEmbeddings

from modular_rag.config import CHAT_MODEL, EMBEDDING_MODEL, OLLAMA_BASE_URL

logger = logging.getLogger(__name__)


def load_models():
    logger.info("Loading Ollama models")
    embeddings_model = OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )
    chat_model = ChatOllama(
        model=CHAT_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.0,
        num_ctx=8192,
    )
    return embeddings_model, chat_model
