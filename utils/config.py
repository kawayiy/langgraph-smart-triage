# config.py
import os

from dotenv import load_dotenv

# 最先加载项目根目录下的 .env，保证后续 os.getenv 能读到环境变量
_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(dotenv_path=_env_path)


class Config:
    """统一的配置类，集中管理所有常量"""
    # prompt文件路径
    PROMPT_TEMPLATE_TXT_AGENT = "prompts/prompt_template_agent.txt"
    PROMPT_TEMPLATE_TXT_GRADE = "prompts/prompt_template_grade.txt"
    PROMPT_TEMPLATE_TXT_REWRITE = "prompts/prompt_template_rewrite.txt"
    PROMPT_TEMPLATE_TXT_GENERATE = "prompts/prompt_template_generate.txt"

    # Chroma 数据库配置
    CHROMADB_DIRECTORY = "chromaDB"
    CHROMADB_COLLECTION_NAME = "demo001"

    # 日志持久化存储
    LOG_FILE = "output/app.log"
    MAX_BYTES=5*1024*1024,
    BACKUP_COUNT=3

    # 数据库 URI，默认值为占位；生产环境必须通过环境变量 DB_URI 注入真实连接串
    DB_URI = os.getenv("DB_URI", "postgresql://user:password@localhost:5432/postgres?sslmode=disable")

    # openai:调用gpt模型, qwen:调用阿里通义千问大模型, oneapi:调用oneapi方案支持的模型, ollama:调用本地开源大模型
    LLM_TYPE = "qwen"

    # API服务地址和端口
    HOST = "0.0.0.0"
    PORT = 8013