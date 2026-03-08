from langchain_chroma import Chroma
from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import tool
from utils.config import Config


def get_tools(llm_embedding):
    """
    Create and return the tool list.

    Args:
        llm_embedding: Embedding model instance used to initialize the vector store.

    Returns:
        list: Tool list.
    """

    # Create the Chroma vector store instance
    vectorstore = Chroma(
        persist_directory=Config.CHROMADB_DIRECTORY,
        collection_name=Config.CHROMADB_COLLECTION_NAME,
        embedding_function=llm_embedding,
    )
    # Convert the vector store into a retriever
    retriever = vectorstore.as_retriever()
    # Create the retrieval tool
    retriever_tool = create_retriever_tool(
        retriever,
        name="retrieve",
        description="这是健康档案查询工具，搜索并返回有关用户的健康档案信息。"
    )


    # Custom multiply tool
    @tool
    def multiply(a: float, b: float) -> float:
        """这是计算两个数的乘积的工具，返回最终的计算结果"""
        return a * b


    # Return the tool list
    return [retriever_tool, multiply]