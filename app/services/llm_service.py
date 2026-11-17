
import logging

from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from app.core.config import settings

logger = logging.getLogger(__name__)

def create_llm() -> ChatOllama:
    """Initialize the Ollama chat model."""

    logger.info(
        "Initializing Ollama model: %s",
        settings.ollama_model,
    )

    model = ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key,
        temperature=0,
    )

    logger.info("Groq model initialized successfully.")

    return model
