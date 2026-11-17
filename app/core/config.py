
import os
from dataclasses import dataclass
from dotenv import load_dotenv
load_dotenv()

@dataclass(frozen=True)
class Settings:
    app_name: str
    environment: str
    log_level: str

    tavily_api_key: str

    ollama_model: str
    ollama_base_url: str

    groq_model: str
    groq_api_key: str

    max_graph_steps: int
    llm_timeout: int
    max_retries: int

    @classmethod
    def load(cls):
        """Load and validate application configuration."""

        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            raise ValueError("TAVILY_API_KEY is missing.")

        return cls(
            app_name=os.getenv("APP_NAME", "tavily-agent"),
            environment=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),

            tavily_api_key=tavily_api_key,

            ollama_model=os.getenv(
                "OLLAMA_MODEL", "qwen2.5:7b"
            ),
            ollama_base_url=os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434"
            ),
            groq_model=os.getenv(
                "GROQ_MODEL"
            ),
            groq_api_key=os.getenv(
                "GROQ_API_KEY"
            ),

            max_graph_steps=int(
                os.getenv("MAX_GRAPH_STEPS", "12")
            ),
            llm_timeout=int(
                os.getenv("LLM_TIMEOUT", "60")
            ),
            max_retries=int(
                os.getenv("MAX_RETRIES", "3")
            ),
        )


settings = Settings.load()
