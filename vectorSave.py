# Purpose: compute PDF embeddings and persist them to Chroma
import os
import logging
from openai import OpenAI
import chromadb
import uuid
from utils import pdfSplitTest_Ch
from utils import pdfSplitTest_En

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# OpenAI model configuration
OPENAI_API_BASE = os.getenv("OPENAI_BASE_URL")
OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# OneAPI model configuration, using Qwen as an example
ONEAPI_API_BASE = os.getenv("ONEAPI_API_BASE", "")  # Configure this in `.env` when using OneAPI
ONEAPI_EMBEDDING_API_KEY = os.getenv("ONEAPI_API_KEY")
ONEAPI_EMBEDDING_MODEL = "text-embedding-v1"

# Alibaba Qwen model configuration
QWen_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWen_EMBEDDING_API_KEY = os.getenv("DASHSCOPE_API_KEY")
QWen_EMBEDDING_MODEL = "text-embedding-v1"

# Local open-source model via vLLM
# Local open-source model via Xinference
# Local open-source model via Ollama, using `bge-m3` as an example
OLLAMA_API_BASE = "http://localhost:11434/v1"
OLLAMA_EMBEDDING_API_KEY = "ollama"
OLLAMA_EMBEDDING_MODEL = "bge-m3:latest"


# `openai`: GPT model, `qwen`: Alibaba Qwen, `oneapi`: models exposed through OneAPI, `ollama`: local open-source models
llmType = "qwen"

# Set the test text language: `Chinese` or `English`
TEXT_LANGUAGE = 'Chinese'
INPUT_PDF = "input/健康档案.pdf"
# TEXT_LANGUAGE = 'English'
# INPUT_PDF = "input/deepseek-v3-1-4.pdf"

# Specify the pages to process; use `None` for all pages
PAGE_NUMBERS=None
# PAGE_NUMBERS=[2, 3]

# Configure the ChromaDB storage path and collection name to fit your environment
CHROMADB_DIRECTORY = "chromaDB"  # Persistent path for the ChromaDB vector store
CHROMADB_COLLECTION_NAME = "demo001"  # Collection name in the target ChromaDB store


# Generate embeddings with `get_embeddings`
def get_embeddings(texts):
    global llmType
    global ONEAPI_API_BASE, ONEAPI_EMBEDDING_API_KEY, ONEAPI_EMBEDDING_MODEL
    global OPENAI_API_BASE, OPENAI_EMBEDDING_API_KEY, OPENAI_EMBEDDING_MODEL
    global QWen_API_BASE, QWen_EMBEDDING_API_KEY, QWen_EMBEDDING_MODEL
    global OLLAMA_API_BASE, OLLAMA_EMBEDDING_API_KEY, OLLAMA_EMBEDDING_MODEL
    if llmType == 'oneapi':
        if not ONEAPI_EMBEDDING_API_KEY:
            logger.error("ONEAPI_API_KEY 未设置，请在环境变量中配置")
            raise ValueError("使用 OneAPI 嵌入时需设置环境变量 ONEAPI_API_KEY")
        try:
            client = OpenAI(
                base_url=ONEAPI_API_BASE,
                api_key=ONEAPI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=ONEAPI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    elif llmType == 'qwen':
        try:
            client = OpenAI(
                base_url=QWen_API_BASE,
                api_key=QWen_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=QWen_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    elif llmType == 'ollama':
        try:
            client = OpenAI(
                base_url=OLLAMA_API_BASE,
                api_key=OLLAMA_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=OLLAMA_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []
    else:
        try:
            client = OpenAI(
                base_url=OPENAI_API_BASE,
                api_key=OPENAI_EMBEDDING_API_KEY
            )
            data = client.embeddings.create(input=texts,model=OPENAI_EMBEDDING_MODEL).data
            return [x.embedding for x in data]
        except Exception as e:
            logger.info(f"生成向量时出错: {e}")
            return []


# Generate embeddings in batches
def generate_vectors(data, max_batch_size=25):
    results = []
    for i in range(0, len(data), max_batch_size):
        batch = data[i:i + max_batch_size]
        # Call `get_embeddings`; the backend depends on the selected API
        response = get_embeddings(batch)
        results.extend(response)
    return results


# Wrapper around ChromaDB with add and search helpers
class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        # Use the global storage directory configuration
        global CHROMADB_DIRECTORY
        # Create a persistent Chroma client under the configured directory
        chroma_client = chromadb.PersistentClient(path=CHROMADB_DIRECTORY)
        # Get an existing collection or create it if it does not exist
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name)
        # Embedding function
        self.embedding_fn = embedding_fn

    # Add documents to the collection
    # Each document includes text and its vector representation for later search
    def add_documents(self, documents):
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # Compute embeddings for the document text
            documents=documents,  # Raw document text
            ids=[str(uuid.uuid4()) for i in range(len(documents))]  # Auto-generated unique document IDs
        )
        
    # Query the vector store and return the most similar matches
    # `query`: query text
    # `top_n`: number of nearest results to return
    def search(self, query, top_n):
        try:
            results = self.collection.query(
                # Embed the query text and run similarity search in the vector store
                query_embeddings=self.embedding_fn([query]),
                n_results=top_n
            )
            return results
        except Exception as e:
            logger.info(f"检索向量数据库时出错: {e}")
            return []


# Wrap text preprocessing and vector-store ingestion for external use
def vectorStoreSave():
    global TEXT_LANGUAGE, CHROMADB_COLLECTION_NAME, INPUT_PDF, PAGE_NUMBERS

    # Test Chinese text
    if TEXT_LANGUAGE == 'Chinese':
        # 1. Get the processed text data
        # This demo processes the selected pages and returns paragraph chunks
        paragraphs = pdfSplitTest_Ch.getParagraphs(
            filename=INPUT_PDF,
            page_numbers=PAGE_NUMBERS,
            min_line_length=1
        )
        # 2. Ingest text chunks into the vector database
        # `collection_name` is the collection name and `embedding_fn` generates embeddings
        vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)
        # Add documents and their embeddings to the vector database
        vector_db.add_documents(paragraphs)
        # 3. Run a retrieval test through the wrapped search interface
        user_query = "张三九的基本信息是什么"
        # Retrieve the top 5 similar results
        search_results = vector_db.search(user_query, 5)
        logger.info(f"检索向量数据库的结果: {search_results}")

    # Test English text
    elif TEXT_LANGUAGE == 'English':
        # 1. Get the processed text data
        # This demo processes the selected pages and returns paragraph chunks
        paragraphs = pdfSplitTest_En.getParagraphs(
            filename=INPUT_PDF,
            page_numbers=PAGE_NUMBERS,
            min_line_length=1
        )
        # 2. Ingest text chunks into the vector database
        # `collection_name` is the collection name and `embedding_fn` generates embeddings
        vector_db = MyVectorDBConnector(CHROMADB_COLLECTION_NAME, generate_vectors)
        # Add documents and their embeddings to the vector database
        vector_db.add_documents(paragraphs)
        # 3. Run a retrieval test through the wrapped search interface
        user_query = "deepseek V3有多少参数"
        # Retrieve the top 5 similar results
        search_results = vector_db.search(user_query, 5)
        logger.info(f"检索向量数据库的结果: {search_results}")


if __name__ == "__main__":
    # Test text preprocessing and vector-store ingestion
    vectorStoreSave()

