# config.py
import os

from dotenv import load_dotenv

# Load the root `.env` file first so later `os.getenv` calls can read the variables
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_env_path)


class Config:
    """Centralized configuration class for project constants."""
    # Prompt template paths
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    # Chroma database configuration
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"

    # Persistent log storage
    LOG_FILE = "output/app.log"
    MAX_BYTES=5*1024*1024,
    BACKUP_COUNT=3

    # Database URI. The default is a placeholder; production must inject the real value through `DB_URI`
    DB_URI = os.getenv("DB_URI", "postgresql://user:password@localhost:5432/postgres?sslmode=disable")

    # `openai`: GPT model, `qwen`: Alibaba Qwen, `oneapi`: models exposed through OneAPI, `ollama`: local open-source models
    LLM_TYPE = "qwen"

    # API host and port
    HOST = "0.0.0.0"
    PORT = 8013