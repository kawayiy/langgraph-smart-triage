# Import the logging module for runtime logs
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
# Import OS interfaces for paths and environment variables
import os
# Import the system module for system-level operations such as exiting
import sys
import threading
import time
# Import the UUID module for generating unique identifiers
import uuid
# Import `escape` from `html` to escape HTML special characters
from html import escape
# Import typing helpers
from typing import Literal, Annotated, Sequence, Optional
# Import `TypedDict` from `typing_extensions` for typed dictionaries
from typing_extensions import TypedDict
# Import LangChain prompt template classes
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
# Import the base LangChain message class
from langchain_core.messages import BaseMessage
# Import the message helper used for appending messages
from langgraph.graph.message import add_messages
# Import the prebuilt tool condition and tool node
from langgraph.prebuilt import tools_condition, ToolNode
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import ToolMessage
# Import the state graph and start/end node constants
from langgraph.graph import StateGraph, START, END
# Import the base store interface
from langgraph.store.base import BaseStore
# Import the runnable configuration class
from langchain_core.runnables import RunnableConfig
# Import the Postgres store implementation
from langgraph.store.postgres import PostgresStore
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# Import psycopg2 operational errors for database connection failures
from psycopg2 import OperationalError
# Import the Postgres checkpoint saver
from langgraph.checkpoint.postgres import PostgresSaver
# Import the PostgreSQL connection pool and timeout exception
from psycopg_pool import ConnectionPool, PoolTimeout
# Import Pydantic base classes and field helpers
from pydantic import BaseModel, Field
# Import the local `get_llm` helper
from utils.llms import get_llm
# Import the tool configuration helper
from utils.tools_config import get_tools
# Import the shared `Config` class
from utils.config import Config

# Configure logging at DEBUG or INFO level
logger = logging.getLogger(__name__)
# Set the logger level to DEBUG
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.INFO)
logger.handlers = []  # Clear default handlers
# Use `ConcurrentRotatingFileHandler`
handler = ConcurrentRotatingFileHandler(
    # Log file
    Config.LOG_FILE,
    # Rotate when the log file reaches 5 MB
    maxBytes = Config.MAX_BYTES,
    # Keep at most 3 historical log files
    backupCount = Config.BACKUP_COUNT
)
# Set the handler level to DEBUG
handler.setLevel(logging.DEBUG)
handler.setFormatter(logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
))
logger.addHandler(handler)


# Message state definition using `TypedDict`
class MessagesState(TypedDict):
    # Message sequence with append behavior handled by `add_messages`
    messages: Annotated[Sequence[BaseMessage], add_messages]
    # Relevance score for retrieved documents
    relevance_score: Annotated[Optional[str], "Relevance score of retrieved documents, 'yes' or 'no'"]
    # Number of query rewrites, used to stop endless graph recursion
    rewrite_count: Annotated[int, "Number of times query has been rewritten"]

# Tool configuration manager for tools and routing rules
class ToolConfig:
    # Initialize the tool list and derived routing metadata
    def __init__(self, tools):
        # Store the incoming tool list
        self.tools = tools
        # Build a set containing all tool names
        self.tool_names = {tool.name for tool in tools}
        # Build the routing map dynamically from the tool definitions
        self.tool_routing_config = self._build_routing_config(tools)
        # Log the initialized tools and routing config for debugging
        logger.info(f"Initialized ToolConfig with tools: {self.tool_names}, routing: {self.tool_routing_config}")

    # Internal helper that builds routing rules from the tool list
    def  _build_routing_config(self, tools):
        # Map tool names to their target graph nodes
        routing_config = {}
        # Process each tool one by one
        for tool in tools:
            # Normalize the tool name to lowercase
            tool_name = tool.name.lower()
            # Treat tools containing `retrieve` as retrieval tools
            if "retrieve" in tool_name:
                # Retrieval tools go to `grade_documents`
                routing_config[tool_name] = "grade_documents"
                # Log the retrieval-tool routing decision
                logger.debug(f"Tool '{tool_name}' routed to 'grade_documents' (retrieval tool)")
            # Non-retrieval tools
            else:
                # Route directly to `generate`
                routing_config[tool_name] = "generate"
                # Log the non-retrieval routing decision
                logger.debug(f"Tool '{tool_name}' routed to 'generate' (non-retrieval tool)")
        # Warn when no routing entries were built
        if not routing_config:
            # The tool list may be empty or malformed
            logger.warning("No tools provided or routing config is empty")
        # Return the generated routing map
        return routing_config

    # Return the tool list
    def get_tools(self):
        # Expose `self.tools`
        return self.tools

    # Return the tool-name set
    def get_tool_names(self):
        # Expose `self.tool_names`
        return self.tool_names

    # Return the tool-routing configuration
    def get_tool_routing_config(self):
        # Expose `self.tool_routing_config`
        return self.tool_routing_config

# Document relevance score schema
class DocumentRelevanceScore(BaseModel):
    # Binary relevance score: `yes` or `no`
    binary_score: str = Field(description="Relevance score 'yes' or 'no'")

# Custom exception for connection-pool initialization or state failures
class ConnectionPoolError(Exception):
    """Custom exception for connection-pool initialization or state failures."""
    pass

# Customized `ToolNode` with concurrent tool execution
class ParallelToolNode(ToolNode):
    # Initialize the node with tools and a max worker count
    def __init__(self, tools, max_workers: int = 5):
        # Initialize the parent `ToolNode`
        super().__init__(tools)
        # Maximum worker count for the thread pool
        self.max_workers = max_workers  # Maximum number of thread-pool workers

    # Execute a single tool call and return a `ToolMessage`
    def _run_single_tool(self, tool_call: dict, tool_map: dict) -> ToolMessage:
        """Execute a single tool call."""
        # Catch tool-execution failures
        try:
            # Read the tool name from the tool call
            tool_name = tool_call["name"]
            # Look up the tool instance
            tool = tool_map.get(tool_name)
            # Fail early if the tool is not registered
            if not tool:
                raise ValueError(f"Tool {tool_name} not found")
            # Execute the tool with its arguments
            result = tool.invoke(tool_call["args"])
            # Return a `ToolMessage` carrying the tool result
            return ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name
            )
        # Log failures and convert them into `ToolMessage` objects
        except Exception as e:
            # Log the tool name and error details
            logger.error(f"Error executing tool {tool_call.get('name', 'unknown')}: {e}")
            # Return an error message back into state
            return ToolMessage(
                content=f"Error: {str(e)}",
                tool_call_id=tool_call["id"],
                name=tool_call.get("name", "unknown")
            )
    '''
    Example
    AIMessage(
    content='',
    tool_calls=[
        {
            "name": "retrieve",
            "args": {"query": "blood pressure record"},
            "id": "call_001"
        },
        {
            "name": "multiply",
            "args": {"a": 2.0, "b": 3.0},
            "id": "call_002"
        }
        ]
    )
    '''

    '''
     Define a callable interface so the instance can be used like a function
     
    After implementing `__call__`, the instance can be invoked like a function:
        node = ParallelToolNode(tools)
        result = node(state)  # Equivalent to `node.__call__(state)`
    '''
    def __call__(self, state: dict) -> dict:
        """Execute all tool calls in parallel."""
        # Log the start of tool-call processing
        logger.info("ParallelToolNode processing tool calls")
        # Read the last message from state
        last_message = state["messages"][-1]
        # Read the tool-call list from the last message
        tool_calls = getattr(last_message, "tool_calls", [])
        # Warn and return early if there are no tool calls
        if not tool_calls:
            logger.warning("No tool calls found in state")
            return {"messages": []}

        # Build a name-to-tool map for quick lookup
        tool_map = {tool.name: tool for tool in self.tools}
        # Collect all returned tool messages here
        results = []

        # Run tool calls concurrently in a thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tool calls and map each future back to its tool call
            future_to_tool = {
                # `executor.submit(callable, arg1, arg2, ...)` forwards args to `_run_single_tool`
                executor.submit(self._run_single_tool, tool_call, tool_map): tool_call
                for tool_call in tool_calls
            }
            # Collect results in completion order
            for future in as_completed(future_to_tool):
                # Catch thread-level execution failures
                try:
                    # Get the returned `ToolMessage`
                    result = future.result()
                    # Append the result
                    results.append(result)
                # Convert unexpected thread failures into error messages
                except Exception as e:
                    # Log the execution failure
                    logger.error(f"Tool execution failed: {e}")
                    # Read the original tool call for this failed future
                    tool_call = future_to_tool[future]
                    # Append an error `ToolMessage`
                    results.append(ToolMessage(
                        content=f"Unexpected error: {str(e)}",
                        tool_call_id=tool_call["id"],
                        name=tool_call.get("name", "unknown")
                    ))

        # Log completion with the number of finished tool calls
        logger.info(f"Completed {len(results)} tool calls")
        # Return the updated state payload
        return {"messages": results}


# Helper for retrieving the latest user question
def get_latest_question(state: MessagesState) -> Optional[str]:
    """Safely retrieve the latest user question from state.

    Args:
        state: Current conversation state containing message history.

    Returns:
        Optional[str]: Latest question content, or `None` if unavailable.
    """
    try:
        # Validate that state contains a non-empty message list
        if not state.get("messages") or not isinstance(state["messages"], (list, tuple)) or len(state["messages"]) == 0:
            logger.warning("No valid messages found in state for getting latest question")
            return None

        # Walk backward to find the most recent `HumanMessage`
        for message in reversed(state["messages"]):
            if message.__class__.__name__ == "HumanMessage" and hasattr(message, "content"):
                return message.content

        # Return `None` if no `HumanMessage` is found
        logger.info("No HumanMessage found in state")
        return None

    except Exception as e:
        logger.error(f"Error getting latest question: {e}")
        return None


# Filter messages kept in thread-local persistence
def filter_messages(messages: list) -> list:
    """Filter messages and keep only `AIMessage` and `HumanMessage` entries."""
    # Keep only `AIMessage` and `HumanMessage` instances
    filtered = [msg for msg in messages if msg.__class__.__name__ in ['AIMessage', 'HumanMessage']]
    # Keep the last N messages when the filtered list grows too large
    return filtered[-5:] if len(filtered) > 5 else filtered


# Store and retrieve cross-thread memory
def store_memory(question: BaseMessage, config: RunnableConfig, store: BaseStore) -> str:
    """Store memory-related information from user input.

    Args:
        question: User input message.
        config: Runtime configuration.
        store: Data store instance.

    Returns:
        str: Memory string related to the current user.
    """
    namespace = ("memories", config["configurable"]["user_id"])
    try:
        # Normalize to string to support multimodal messages where `content` is a list
        content_str = question.content if isinstance(question.content, str) else str(question.content)
        # Search related memories and cap them at 10 to avoid excessive context
        raw_memories = store.search(namespace, query=content_str)
        memories = list(raw_memories)[:10] if raw_memories is not None else []
        # Read `value["data"]` safely to support older or incomplete records
        parts = []
        for d in memories:
            val = d.value if d.value is not None else {}
            text = val.get("data", "")
            if text:
                parts.append(text)
        user_info = "\n".join(parts)

        # Store new memory when the user explicitly uses the memory trigger phrase
        if "记住" in content_str:
            memory = escape(content_str)
            store.put(namespace, str(uuid.uuid4()), {"data": memory})
            logger.info(f"Stored memory: {memory}")

        return user_info
    except Exception as e:
        logger.error(f"Error in store_memory: {e}")
        return ""


# Build an LLM processing chain
def create_chain(llm_chat, template_file: str, structured_output=None):
    """Create an LLM chain by loading a prompt template and binding the model.

    Args:
        llm_chat: Language model instance.
        template_file: Prompt template file path.
        structured_output: Optional structured-output schema.

    Returns:
        Runnable: Configured processing chain.

    Raises:
        FileNotFoundError: Raised when the template file is missing.
    """
    # Initialize static cache and lock on first call only
    if not hasattr(create_chain, "prompt_cache"):
        # Prompt cache
        create_chain.prompt_cache = {}
        # Lock used to keep cache access thread-safe
        create_chain.lock = threading.Lock()

    try:
        # Check the cache first without locking
        if template_file in create_chain.prompt_cache:
            prompt_template = create_chain.prompt_cache[template_file]
            logger.info(f"Using cached prompt template for {template_file}")
        else:
            # Protect cache writes with a lock
            with create_chain.lock:
                # Double-check that the template is not already cached
                if template_file not in create_chain.prompt_cache:
                    logger.info(f"Loading and caching prompt template from {template_file}")
                    # Load the template from disk and cache it
                    create_chain.prompt_cache[template_file] = PromptTemplate.from_file(template_file, encoding="utf-8")
                # Read the template from cache
                prompt_template = create_chain.prompt_cache[template_file]

        # Build the chat prompt from the template text
        prompt = ChatPromptTemplate.from_messages([("human", prompt_template.template)])
        # Return the prompt+LLM chain and bind structured output when needed
        return prompt | (llm_chat.with_structured_output(structured_output) if structured_output else llm_chat)
    except FileNotFoundError:
        logger.error(f"Template file {template_file} not found")
        raise


# Retry database operations up to 3 times with exponential backoff from 2 to 10 seconds
@retry(stop=stop_after_attempt(3),wait=wait_exponential(multiplier=1, min=2, max=10),retry=retry_if_exception_type(OperationalError))
def test_connection(db_connection_pool: ConnectionPool) -> bool:
    """Test whether the connection pool is usable."""
    with db_connection_pool.getconn() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            if result != (1,):
                raise ConnectionPoolError("连接池测试查询失败，返回结果异常")
    return True


# Periodically inspect the connection-pool status and emit early warnings
def monitor_connection_pool(db_connection_pool: ConnectionPool, interval: int = 60):
    """Monitor connection-pool status periodically."""
    def _monitor():
        while not db_connection_pool.closed:
            try:
                stats = db_connection_pool.get_stats()
                active = stats.get("connections_in_use", 0)
                total = db_connection_pool.max_size
                logger.info(f"Connection db_connection_pool status: {active}/{total} connections in use")
                if active >= total * 0.8:
                    logger.warning(f"Connection db_connection_pool nearing capacity: {active}/{total}")
            except Exception as e:
                logger.error(f"Failed to monitor connection db_connection_pool: {e}")
            time.sleep(interval)

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()
    return monitor_thread


# Agent node used for routing/triage
def agent(state: MessagesState, config: RunnableConfig, *, store: BaseStore, llm_chat, tool_config: ToolConfig) -> dict:
    """Agent function that decides whether to call tools or finish.

    Args:
        state: Current conversation state.
        config: Runtime configuration.
        store: Data store instance.
        llm_chat: Chat model.
        tool_config: Tool configuration.

    Returns:
        dict: Updated conversation state.
    """
    # Log the start of agent processing
    logger.info("Agent processing user query")
    # Use the user ID as the memory namespace
    namespace = ("memories", config["configurable"]["user_id"])
    # Execute the agent logic
    try:
        # Read the latest message as the user question
        question = state["messages"][-1]
        logger.info(f"agent question:{question}")

        # Read and update cross-thread memory
        user_info = store_memory(question, config, store)
        # Filter the in-thread message history
        messages = filter_messages(state["messages"])

        # Bind tools to the LLM
        llm_chat_with_tool = llm_chat.bind_tools(tool_config.get_tools())

        # Create the agent chain
        agent_chain = create_chain(llm_chat_with_tool, Config.PROMPT_TEMPLATE_TXT_AGENT)
        # Invoke the agent chain
        response = agent_chain.invoke({"question": question,"messages": messages, "userInfo": user_info})
        # logger.info(f"Agent response: {response}")
        # Return the updated state
        return {"messages": [response]}
    # Catch unexpected failures
    except Exception as e:
        # Log the failure
        logger.error(f"Error in agent processing: {e}")
        # Return an error message
        return {"messages": [{"role": "system", "content": "处理请求时出错"}]}


# Node that scores document relevance
def grade_documents(state: MessagesState, llm_chat) -> dict:
    """Score document relevance against the user question and store the result in state.

    Args:
        state: Current conversation state containing message history.

    Returns:
        dict: Updated state containing the score.
    """
    logger.info("Grading documents for relevance")
    if not state.get("messages"):
        logger.error("Messages state is empty")
        return {
            "messages": [{"role": "system", "content": "状态为空，无法评分"}],
            "relevance_score": None
        }

    try:
        # Get the latest user question
        question = get_latest_question(state)
        # Use the latest message content as context because tool output is written there
        context = state["messages"][-1].content
        # logger.info(f"Evaluating relevance - Question: {question}, Context: {context}")

        # Create the relevance-scoring chain
        grade_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GRADE, DocumentRelevanceScore)
        # Run the relevance-scoring chain
        scored_result = grade_chain.invoke({"question": question, "context": context})
        # logger.info(f"scored_result:{scored_result}")
        # Read the score
        score = scored_result.binary_score
        logger.info(f"Document relevance score: {score}")

        # Return the updated state with the score
        return {
            # Keep messages unchanged
            "messages": state["messages"],
            # Store the relevance score
            "relevance_score": score
        }
    except (IndexError, KeyError) as e:
        logger.error(f"Message access error: {e}")
        return {
            "messages": [{"role": "system", "content": "无法评分文档"}],
            "relevance_score": None
        }
    except Exception as e:
        logger.error(f"Unexpected error in grading: {e}")
        return {
            "messages": [{"role": "system", "content": "评分过程中出错"}],
            "relevance_score": None
        }


# Query rewriting
def rewrite(state: MessagesState, llm_chat) -> dict:
    """Rewrite the user query to improve retrieval quality.

    Args:
        state: Current conversation state.

    Returns:
        dict: Updated message state.
    """
    # Log the start of query rewriting
    logger.info("Rewriting query")
    # Execute the rewrite logic
    try:
        # Read the latest user question
        question = get_latest_question(state)
        # Create the rewrite chain
        rewrite_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_REWRITE)
        # Invoke the chain to generate a rewritten query
        response = rewrite_chain.invoke({"question": question})
        # logger.info(f"rewrite question:{response}")
        # Increment the rewrite counter
        rewrite_count = state.get("rewrite_count", 0) + 1
        logger.info(f"Rewrite count: {rewrite_count}")
        # Return the updated state
        return {"messages": [response], "rewrite_count": rewrite_count}
    # Catch index/key access errors
    except (IndexError, KeyError) as e:
        # Log the failure
        logger.error(f"Message access error in rewrite: {e}")
        # Return an error message
        return {"messages": [{"role": "system", "content": "无法重写查询"}]}


# Node that generates the final answer
def generate(state: MessagesState, llm_chat) -> dict:
    """Generate the final reply from tool output.

    Args:
        state: Current conversation state.

    Returns:
        dict: Updated message state.
    """
    # Log the start of answer generation
    logger.info("Generating final response")
    # Execute the generation logic
    try:
        # Read the latest user question
        question = get_latest_question(state)
        # Use the latest message content as context because tool output is written there
        context = state["messages"][-1].content
        # logger.info(f"generate - Question: {question}, Context: {context}")
        # Create the generation chain
        generate_chain = create_chain(llm_chat, Config.PROMPT_TEMPLATE_TXT_GENERATE)
        # Invoke the generation chain
        response = generate_chain.invoke({"context": context, "question": question})
        # Return the updated message state
        return {"messages": [response]}
    # Catch index/key access errors
    except (IndexError, KeyError) as e:
        # Log the failure
        logger.error(f"Message access error in generate: {e}")
        # Return an error message
        return {"messages": [{"role": "system", "content": "无法生成回复"}]}


# Edge router that chooses the next node after tool execution
def route_after_tools(state: MessagesState, tool_config: ToolConfig) -> Literal["generate", "grade_documents"]:
    """
    Choose the next route dynamically based on tool-call results.

    Args:
        state: Current conversation state containing message history and possible tool output.
        tool_config: Tool configuration.

    Returns:
        Literal["generate", "grade_documents"]: Target node for the next step.
    """
    # Default to `generate` when the message list is missing or invalid
    if not state.get("messages") or not isinstance(state["messages"], list):
        logger.error("Messages state is empty or invalid, defaulting to generate")
        return "generate"

    try:
        # Read the last message to identify the tool source
        last_message = state["messages"][-1]

        # Route to `generate` if the message has no `name`
        if not hasattr(last_message, "name") or last_message.name is None:
            logger.info("Last message has no name attribute, routing to generate")
            return "generate"

        # Make sure the message came from a registered tool
        tool_name = last_message.name
        if tool_name not in tool_config.get_tool_names():
            logger.info(f"Unknown tool {tool_name}, routing to generate")
            return "generate"

        # Use the routing config and default to `generate`
        target = tool_config.get_tool_routing_config().get(tool_name, "generate")
        logger.info(f"Tool {tool_name} routed to {target} based on config")
        return target

    except IndexError:
        # Default to `generate` on empty-message/index errors
        logger.error("No messages available in state, defaulting to generate")
        return "generate"
    except AttributeError:
        # Default to `generate` on invalid message objects
        logger.error("Invalid message object, defaulting to generate")
        return "generate"
    except Exception as e:
        # Default to `generate` on any unexpected error
        logger.error(f"Unexpected error in route_after_tools: {e}, defaulting to generate")
        return "generate"


# Edge router that chooses the next node after relevance grading
def route_after_grade(state: MessagesState) -> Literal["generate", "rewrite"]:
    """
    Choose the next route based on the relevance score with extra validation.

    Args:
        state: Current conversation state, expected to contain `messages` and `relevance_score`.

    Returns:
        Literal["generate", "rewrite"]: Target node for the next step.
    """
    # Default to `rewrite` when state is not a valid dictionary
    if not isinstance(state, dict):
        logger.error("State is not a valid dictionary, defaulting to rewrite")
        return "rewrite"

    # Default to `rewrite` when `messages` is missing or invalid
    if "messages" not in state or not isinstance(state["messages"], (list, tuple)):
        logger.error("State missing valid messages field, defaulting to rewrite")
        return "rewrite"

    # Default to `rewrite` when the message list is empty
    if not state["messages"]:
        logger.warning("Messages list is empty, defaulting to rewrite")
        return "rewrite"

    # Read the relevance score, defaulting to `None`
    relevance_score = state.get("relevance_score")
    # Read the rewrite counter
    rewrite_count = state.get("rewrite_count", 0)
    logger.info(f"Routing based on relevance_score: {relevance_score}, rewrite_count: {rewrite_count}")

    # Force `generate` after 3 rewrites
    if rewrite_count >= 3:
        logger.info("Max rewrite limit reached, proceeding to generate")
        return "generate"

    try:
        # Treat non-string relevance scores as invalid
        if not isinstance(relevance_score, str):
            logger.warning(f"Invalid relevance_score type: {type(relevance_score)}, defaulting to rewrite")
            return "rewrite"

        # Route to `generate` when the score is `yes`
        if relevance_score.lower() == "yes":
            logger.info("Documents are relevant, proceeding to generate")
            return "generate"

        # Route to `rewrite` for `no` or any other value
        logger.info("Documents are not relevant or scoring failed, proceeding to rewrite")
        return "rewrite"

    except AttributeError:
        # Default to `rewrite` when `lower()` is not supported
        logger.error("relevance_score is not a string or is None, defaulting to rewrite")
        return "rewrite"
    except Exception as e:
        # Default to `rewrite` on any unexpected error
        logger.error(f"Unexpected error in route_after_grade: {e}, defaulting to rewrite")
        return "rewrite"


# Save a visualization of the state graph
def save_graph_visualization(graph: StateGraph, filename: str = "graph.png") -> None:
    """Save a visualization of the state graph.

    Args:
        graph: State graph instance.
        filename: Output file path.
    """
    # Try to render and save the graph visualization
    try:
        # Open the target file in binary-write mode
        with open(filename, "wb") as f:
            # Render the graph as a Mermaid PNG and write it to disk
            f.write(graph.get_graph().draw_mermaid_png())
        # Log save success
        logger.info(f"Graph visualization saved as {filename}")
    # Catch file I/O errors
    except IOError as e:
        # Log a warning instead of failing hard
        logger.warning(f"Failed to save graph visualization: {e}")


# Create and configure the state graph
def create_graph(db_connection_pool: ConnectionPool, llm_chat, llm_embedding, tool_config: ToolConfig) -> StateGraph:
    """Create and configure the state graph.

    Args:
        db_connection_pool: Database connection pool.
        llm_chat: Chat model.
        llm_embedding: Embedding model.
        tool_config: Tool configuration.

    Returns:
        StateGraph: Compiled state graph.

    Raises:
        ConnectionPoolError: Raised when the connection pool is unavailable or invalid.
    """
    # Validate that the connection pool exists and is open
    if db_connection_pool is None or db_connection_pool.closed:
        logger.error("Connection db_connection_pool is None or closed")
        raise ConnectionPoolError("数据库连接池未初始化或已关闭")
    try:
        # Read the current and maximum connection counts
        active_connections = db_connection_pool.get_stats().get("connections_in_use", 0)
        max_connections = db_connection_pool.max_size
        if active_connections >= max_connections:
            logger.error(f"Connection db_connection_pool exhausted: {active_connections}/{max_connections} connections in use")
            raise ConnectionPoolError("连接池已耗尽，无可用连接")
        if not test_connection(db_connection_pool):
            raise ConnectionPoolError("连接池测试失败")
        logger.info("Connection db_connection_pool status: OK, test connection successful")
    except PoolTimeout as e:
        logger.error(f"Connection pool timeout: {e}")
        raise ConnectionPoolError(
            "无法在超时时间内从连接池获取连接。请确认：1) PostgreSQL 服务已启动；"
            "2) .env 中 DB_URI 的主机、端口、用户名、密码正确；3) 若使用 Docker，容器已运行且端口已映射。"
        ) from e
    except OperationalError as e:
        logger.error(f"Database operational error during connection test: {e}")
        raise ConnectionPoolError(f"连接池测试失败，可能已关闭或超时: {str(e)}") from e
    except Exception as e:
        logger.error(f"Failed to verify connection db_connection_pool status: {e}")
        raise ConnectionPoolError(f"无法验证连接池状态: {str(e)}") from e

    # In-thread persistence
    try:
        # Create the Postgres checkpoint saver for short-term memory
        checkpointer = PostgresSaver(db_connection_pool)
        # Initialize checkpoints
        checkpointer.setup()
    except Exception as e:
        logger.error(f"Failed to setup PostgresSaver: {e}")
        raise ConnectionPoolError(f"检查点初始化失败: {str(e)}")

    # Cross-thread persistence
    try:
        # Create the Postgres store for long-term memory
        store = PostgresStore(db_connection_pool, index={"dims": 1536, "embed": llm_embedding})
        store.setup()
    except Exception as e:
        logger.error(f"Failed to setup PostgresStore: {e}")
        raise ConnectionPoolError(f"存储初始化失败: {str(e)}")

    # Create the workflow graph with `MessagesState`
    workflow = StateGraph(MessagesState)
    # Add the agent node
    workflow.add_node("agent", lambda state, config: agent(state, config, store=store, llm_chat=llm_chat, tool_config=tool_config))
    # Add the parallel tool node
    workflow.add_node("call_tools", ParallelToolNode(tool_config.get_tools(), max_workers=5))
    # Add the rewrite node
    workflow.add_node("rewrite", lambda state: rewrite(state,llm_chat=llm_chat))
    # Add the generate node
    workflow.add_node("generate", lambda state: generate(state, llm_chat=llm_chat))
    # Add the document-relevance grading node
    workflow.add_node("grade_documents", lambda state: grade_documents(state, llm_chat=llm_chat))

    # Add the edge from `START` to `agent`
    workflow.add_edge(START, end_key="agent")
    # Add conditional edges after `agent` based on tool usage
    workflow.add_conditional_edges(source="agent", path=tools_condition, path_map={"tools": "call_tools", END: END})
    # Add conditional edges after tool execution
    workflow.add_conditional_edges(source="call_tools", path=lambda state: route_after_tools(state, tool_config),path_map={"generate": "generate", "grade_documents": "grade_documents"})
    # Add conditional edges after relevance grading
    workflow.add_conditional_edges(source="grade_documents", path=route_after_grade, path_map={"generate": "generate", "rewrite": "rewrite"})
    # Add the edge from `generate` to `END`
    workflow.add_edge(start_key="generate", end_key=END)
    # Add the edge from `rewrite` back to `agent`
    workflow.add_edge(start_key="rewrite", end_key="agent")

    # Compile the graph with checkpointing and storage
    return workflow.compile(checkpointer=checkpointer, store=store)


# Response helper for CLI/console usage
def graph_response(graph: StateGraph, user_input: str, config: dict, tool_config: ToolConfig) -> None:
    """Process user input and print either tool output or LLM output.

    Args:
        graph: State graph instance.
        user_input: User input text.
        config: Runtime configuration.
    """
    try:
        # Start streaming events from the graph
        events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        # Iterate through the event stream
        for event in events:
            # Iterate through event values
            for value in event.values():
                # Skip invalid message payloads
                if "messages" not in value or not isinstance(value["messages"], list):
                    logger.warning("No valid messages in response")
                    continue

                # Read the last message
                last_message = value["messages"][-1]

                # Detect tool calls
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    # Iterate over tool calls
                    for tool_call in last_message.tool_calls:
                        # Validate the tool-call payload
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            # Log the tool call
                            logger.info(f"Calling tool: {tool_call['name']}")
                    # Skip to the next event
                    continue

                # Check whether the message has content
                if hasattr(last_message, "content"):
                    content = last_message.content

                    # Case 1: tool output
                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        tool_name = last_message.name
                        print(f"Tool Output [{tool_name}]: {content}")
                    # Case 2: normal LLM output
                    else:
                        print(f"Assistant: {content}")
                else:
                    # Messages with no content may represent intermediate state
                    logger.info("Message has no content, skipping")
                    print("Assistant: 未获取到相关回复")
    except ValueError as ve:
        logger.error(f"Value error in response processing: {ve}")
        print("Assistant: 处理响应时发生值错误")
    except Exception as e:
        logger.error(f"Error processing response: {e}")
        print("Assistant: 处理响应时发生未知错误")


# Main entry point
def main():
    """Initialize and run the chatbot."""
    # Initialize the connection pool as `None`
    db_connection_pool = None
    try:
        # Initialize the chat and embedding models
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

        # Get the tool list
        tools = get_tools(llm_embedding)

        # Create the `ToolConfig` instance
        tool_config = ToolConfig(tools)

        # Connection settings: autocommit, no prepare threshold, 5-second connect timeout
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 5}
        # Create the connection pool with max 20, min 2, and a 10-second wait timeout
        db_connection_pool = ConnectionPool(conninfo=Config.DB_URI, max_size=20, min_size=2, kwargs=connection_kwargs, timeout=10)

        # Open the connection pool
        try:
            db_connection_pool.open()
            logger.info("Database connection pool initialized")
            logger.debug("Database connection pool initialized")
        except Exception as e:
            logger.error(f"Failed to open connection pool: {e}")
            raise ConnectionPoolError(f"无法打开数据库连接池: {str(e)}")

        # Start the connection-pool monitor as a daemon thread
        monitor_thread = monitor_connection_pool(db_connection_pool, interval=60)

        # Create the state graph
        try:
            graph = create_graph(db_connection_pool, llm_chat, llm_embedding, tool_config)
        except ConnectionPoolError as e:
            logger.error(f"Graph creation failed: {e}")
            print(f"错误: {e}")
            sys.exit(1)

        # Save the graph visualization
        save_graph_visualization(graph)

        # Print the ready message
        print("聊天机器人准备就绪！输入 'quit'、'exit' 或 'q' 结束对话。")
        # Runtime configuration with thread and user IDs
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}
        # Enter the main loop
        while True:
            # Read user input and trim whitespace
            user_input = input("User: ").strip()
            # Exit when the user enters a quit command
            if user_input.lower() in {"quit", "exit", "q"}:
                print("拜拜!")
                break
            # Skip empty input
            if not user_input:
                print("请输入聊天内容！")
                continue
            # Process the user input and stream/print the response
            graph_response(graph, user_input, config, tool_config)

    except ConnectionPoolError as e:
        # Catch connection-pool-related errors
        logger.error(f"Connection pool error: {e}")
        print(f"错误: 数据库连接池问题 - {e}")
        sys.exit(1)
    except RuntimeError as e:
        # Catch other runtime errors
        logger.error(f"Initialization error: {e}")
        print(f"错误: 初始化失败 - {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        # Catch keyboard interruption
        print("\n被用户打断。再见！")
    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Unexpected error: {e}")
        print(f"错误: 发生未知错误 - {e}")
        sys.exit(1)
    finally:
        # Clean up resources
        if db_connection_pool and not db_connection_pool.closed:
            db_connection_pool.close()
            logger.info("Database connection pool closed")


# Run `main()` when executed as the entry module
if __name__ == "__main__":
    # Call the main function
    main()