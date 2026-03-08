import os
import logging
from langchain_openai import ChatOpenAI,OpenAIEmbeddings


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model configuration mapping
MODEL_CONFIGS = {
    "openai": {
        "base_url": os.getenv("OPENAI_BASE_URL"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "chat_model": "gpt-4o",
        "embedding_model": "text-embedding-3-small"
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1"
    },
    "oneapi": {
        "base_url": os.getenv("ONEAPI_API_BASE", ""),  # Configure `ONEAPI_API_BASE` in `.env` when using OneAPI
        "api_key": os.getenv("ONEAPI_API_KEY"),
        "chat_model": "qwen-max",
        "embedding_model": "text-embedding-v1"
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "chat_model": "qwen2.5:32b",
        "embedding_model": "bge-m3:latest"
    }
}


# Default settings
DEFAULT_LLM_TYPE = "qwen"
DEFAULT_TEMPERATURE = 0.0


class LLMInitializationError(Exception):
    """Custom exception raised for LLM initialization errors."""
    pass


def initialize_llm(llm_type: str = DEFAULT_LLM_TYPE) -> tuple[ChatOpenAI, OpenAIEmbeddings]:
    """
    Initialize an LLM instance.

    Args:
        llm_type (str): LLM type. Supported values: `openai`, `oneapi`, `qwen`, `ollama`.

    Returns:
        ChatOpenAI: Initialized chat model instance.

    Raises:
        LLMInitializationError: Raised when LLM initialization fails.
    """
    try:
        # Validate the requested LLM type
        if llm_type not in MODEL_CONFIGS:
            raise ValueError(f"不支持的LLM类型: {llm_type}. 可用的类型: {list(MODEL_CONFIGS.keys())}")

        config = MODEL_CONFIGS[llm_type]

        # Apply Ollama-specific handling
        if llm_type == "ollama":
            os.environ["OPENAI_API_KEY"] = "NA"

        # Create the chat model instance
        llm_chat = ChatOpenAI(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["chat_model"],
            temperature=DEFAULT_TEMPERATURE,
            timeout=30,  # Request timeout in seconds
            max_retries=2  # Number of retries
        )

        llm_embedding = OpenAIEmbeddings(
            base_url=config["base_url"],
            api_key=config["api_key"],
            model=config["embedding_model"],
            deployment=config["embedding_model"],
            check_embedding_ctx_length=False
        )

        logger.info(f"成功初始化 {llm_type} LLM")
        return llm_chat, llm_embedding

    except ValueError as ve:
        logger.error(f"LLM配置错误: {str(ve)}")
        raise LLMInitializationError(f"LLM配置错误: {str(ve)}")
    except Exception as e:
        logger.error(f"初始化LLM失败: {str(e)}")
        raise LLMInitializationError(f"初始化LLM失败: {str(e)}")


def get_llm(llm_type: str = DEFAULT_LLM_TYPE) -> ChatOpenAI:
    """
    Wrapper for retrieving an LLM instance with defaults and error handling.

    Args:
        llm_type (str): LLM type.

    Returns:
        ChatOpenAI: LLM instance.
    """
    try:
        return initialize_llm(llm_type)
    except LLMInitializationError as e:
        logger.warning(f"使用默认配置重试: {str(e)}")
        if llm_type != DEFAULT_LLM_TYPE:
            return initialize_llm(DEFAULT_LLM_TYPE)
        raise  # Re-raise if the default configuration also fails


# Example usage
if __name__ == "__main__":
    try:
        # Test initialization for different LLM types
        # llm_openai = get_llm("openai")
        llm_qwen = get_llm("qwen")

        # Test an invalid type
        # llm_invalid = get_llm("invalid_type")
    except LLMInitializationError as e:
        logger.error(f"程序终止: {str(e)}")