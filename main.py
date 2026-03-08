# Import OS interfaces for paths and environment variables
import os
# Used for regular-expression matching and string processing
import re
# Used for JSON serialization and deserialization
import json
# Used to define an asynchronous context manager
from contextlib import asynccontextmanager
# Used for type hints such as lists and optional values
from typing import List, Tuple
# Used to create the web app and handle HTTP exceptions
from fastapi import FastAPI, HTTPException, Depends
# Used to return JSON and streaming responses
from fastapi.responses import JSONResponse, StreamingResponse
# Used to run the FastAPI application
import uvicorn
# Import the logging module for runtime logs
import logging
from concurrent_log_handler import ConcurrentRotatingFileHandler
# Import the system module for system-level operations such as exiting
import sys
import time
# Import the UUID module for generating unique identifiers
import uuid
# Import typing utilities
from typing import Optional
# Import Pydantic base classes and field helpers
from pydantic import BaseModel, Field
# Import helpers from the local project
from ragAgent import (
    ToolConfig,
    create_graph,
    save_graph_visualization,
    get_llm,
    get_tools,
    Config,
    ConnectionPool,
    ConnectionPoolError,
    monitor_connection_pool,
)


# Optional LangSmith environment variables for tracing application steps
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = ""


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


# Message model used to wrap API response data
# Define the `Message` class
class Message(BaseModel):
    role: str
    content: str

# Define the `ChatCompletionRequest` class
class ChatCompletionRequest(BaseModel):
    messages: List[Message]
    stream: Optional[bool] = False
    userId: Optional[str] = None
    conversationId: Optional[str] = None

# Define the `ChatCompletionResponseChoice` class
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None

# Define the `ChatCompletionResponse` class
class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    system_fingerprint: Optional[str] = None


def format_response(response):
    """Split the input text into paragraphs, add line breaks, and preserve code blocks for better readability.

    Args:
        response: Input text.

    Returns:
        Text with clearer paragraph separation.
    """
    # Split the input on two or more consecutive newlines to form paragraphs
    paragraphs = re.split(r'\n{2,}', response)
    # List used to collect formatted paragraphs
    formatted_paragraphs = []
    # Process each paragraph one by one
    for para in paragraphs:
        # Check whether the paragraph contains code fences
        if '```' in para:
            # Split on ``` so code blocks and plain text alternate
            parts = para.split('```')
            for i, part in enumerate(parts):
                # Odd-numbered parts represent code blocks
                if i % 2 == 1:  # This part is a code block
                    # Wrap the code block with line breaks and trim extra whitespace
                    parts[i] = f"\n```\n{part.strip()}\n```\n"
            # Recombine the split parts into a single string
            para = ''.join(parts)
        else:
            # Otherwise, replace ". " with a newline to separate sentences
            para = para.replace('. ', '.\n')
        # Append the cleaned paragraph to `formatted_paragraphs`
        # `strip()` removes leading and trailing whitespace characters
        formatted_paragraphs.append(para.strip())
    # Join formatted paragraphs with blank lines between them
    return '\n\n'.join(formatted_paragraphs)


# Async context manager for startup and shutdown initialization/cleanup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage FastAPI application startup and shutdown tasks.

    Args:
        app (FastAPI): FastAPI application instance.

    Yields:
        None: Initialization happens before `yield`, cleanup after it.

    Raises:
        ConnectionPoolError: Raised when connection-pool initialization or operations fail.
        Exception: Raised for any other unexpected errors.
    """
    # Declare the shared `graph` and `tool_config`
    global graph, tool_config
    # Initialize the database connection pool as `None`
    db_connection_pool = None
    try:
        # Initialize the chat model and embedding model
        llm_chat, llm_embedding = get_llm(Config.LLM_TYPE)

        # Build the tool list from the embedding model
        tools = get_tools(llm_embedding)

        # Create the tool configuration instance
        tool_config = ToolConfig(tools)

        # Database connection settings: autocommit, no prepare threshold, 10-second connect timeout
        connection_kwargs = {"autocommit": True, "prepare_threshold": 0, "connect_timeout": 10}
        # Create a pool with up to 20 connections, at least 2 active, and a 30-second wait timeout
        db_connection_pool = ConnectionPool(
            conninfo=Config.DB_URI,
            max_size=20,
            min_size=2,
            kwargs=connection_kwargs,
            timeout=30
        )

        # Try to open the database connection pool
        try:
            # Open the pool so connections become available
            db_connection_pool.open()
            # Log pool initialization success at INFO level
            logger.info("Database connection pool initialized")
            # Log detailed initialization info at DEBUG level
            logger.debug("Database connection pool initialized")
        except Exception as e:
            # Log a pool-open failure
            logger.error(f"Failed to open connection pool: {e}")
            # Raise the custom connection-pool exception
            raise ConnectionPoolError(f"无法打开数据库连接池: {str(e)}")

        # Start the pool monitor thread and check every 60 seconds
        monitor_thread = monitor_connection_pool(db_connection_pool, interval=60)

        # Try to create the state graph
        try:
            # Create the graph with the database pool and models
            graph = create_graph(db_connection_pool, llm_chat, llm_embedding, tool_config)
        except ConnectionPoolError as e:
            # Log graph-creation failure
            logger.error(f"Graph creation failed: {e}")
            # Exit with status code 1
            sys.exit(1)

        # Save the graph visualization
        save_graph_visualization(graph)

    except ConnectionPoolError as e:
        # Catch and log connection-pool errors
        logger.error(f"Connection pool error: {e}")
        # Exit with status code 1
        sys.exit(1)
    except Exception as e:
        # Catch and log unexpected errors
        logger.error(f"Unexpected error: {e}")
        # Exit with status code 1
        sys.exit(1)

    # `yield` hands control back while the application is running
    yield
    # Close the database connection pool during cleanup
    if db_connection_pool and not db_connection_pool.closed:
        # Close the pool
        db_connection_pool.close()
        # Log pool shutdown
        logger.info("Database connection pool closed")
    # Log service shutdown
    logger.info("The service has been shut down")

# Create the FastAPI app; `lifespan` runs initialization and cleanup hooks
app = FastAPI(lifespan=lifespan)


# Handle non-streaming responses and return the full payload
async def handle_non_stream_response(user_input, graph, tool_config, config):
    """
    Handle a non-streaming response and return the complete payload.

    Args:
        user_input (str): User input text.
        graph: Graph object used to process the message flow.
        tool_config: Tool configuration object containing available tools.
        config (dict): Runtime config containing thread and user identifiers.

    Returns:
        JSONResponse: JSON response containing the formatted output.
    """
    # Hold the final response content
    content = None
    try:
        # Start `graph.stream` to process the user input as an event stream
        events = graph.stream({"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0}, config)
        # Iterate over each event in the stream
        for event in events:
            # Iterate over each value in the event
            for value in event.values():
                # Skip invalid message payloads
                if "messages" not in value or not isinstance(value["messages"], list):
                    # Log and skip invalid messages
                    logger.warning("No valid messages in response")
                    continue

                # Read the last message in the list
                last_message = value["messages"][-1]

                # Detect tool calls in the message
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    # Iterate over all tool calls
                    for tool_call in last_message.tool_calls:
                        # Ensure the tool call is a dict with a name
                        if isinstance(tool_call, dict) and "name" in tool_call:
                            # Log the tool call
                            logger.info(f"Calling tool: {tool_call['name']}")
                    # Skip to the next event
                    continue

                # Check whether the message contains content
                if hasattr(last_message, "content"):
                    # Save the message content
                    content = last_message.content

                    # Detect tool output by checking the tool name
                    if hasattr(last_message, "name") and last_message.name in tool_config.get_tool_names():
                        # Read the tool name
                        tool_name = last_message.name
                        # Log the tool output
                        logger.info(f"Tool Output [{tool_name}]: {content}")
                    # Handle normal LLM output
                    else:
                        # Log the final response
                        logger.info(f"Final Response is: {content}")
                else:
                    # Log and skip messages with no content
                    logger.info("Message has no content, skipping")
    except ValueError as ve:
        # Catch and log value errors
        logger.error(f"Value error in response processing: {ve}")
    except Exception as e:
        # Catch and log unexpected errors
        logger.error(f"Error processing response: {e}")

    # Format the response or fall back to a default value
    formatted_response = str(format_response(content)) if content else "No response generated"
    # Log the formatted response
    logger.info(f"Results for Formatting: {formatted_response}")

    # Build the response object returned to the client
    try:
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content=formatted_response),
                    finish_reason="stop"
                )
            ]
        )
    except Exception as resp_error:
        # Catch and log response-construction errors
        logger.error(f"Error creating response object: {resp_error}")
        # Build a fallback error response
        response = ChatCompletionResponse(
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=Message(role="assistant", content="Error generating response"),
                    finish_reason="error"
                )
            ]
        )

    # Log the response payload sent to the client
    logger.info(f"Send response content: \n{response}")
    # Return the JSON response
    return JSONResponse(content=response.model_dump())


# Handle streaming responses and return SSE output
async def handle_stream_response(user_input, graph, config):
    """
    Handle a streaming response and return SSE output.

    Args:
        user_input (str): User input text.
        graph: Graph object used to process the message flow.
        config (dict): Runtime config containing thread and user identifiers.

    Returns:
        StreamingResponse: Streaming response with media type `text/event-stream`.
    """
    async def generate_stream():
        """
        Internal async generator that yields streaming response data.

        Yields:
            str: Streaming data chunk in SSE (Server-Sent Events) format.

        Raises:
            Exception: Any exception raised during stream generation.
        """
        try:
            # Generate a unique chunk ID
            chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
            # Stream message chunks from the graph
            stream_data = graph.stream(
                {"messages": [{"role": "user", "content": user_input}], "rewrite_count": 0},
                config,
                stream_mode="messages"
            )
            # Iterate over each streamed message chunk
            for message_chunk, metadata in stream_data:
                try:
                    # Read the current node name
                    node_name = metadata.get("langgraph_node") if metadata else None
                    # Only process the `generate` and `agent` nodes
                    if node_name in ["generate", "agent"]:
                        # Read the chunk content, defaulting to an empty string
                        chunk = getattr(message_chunk, 'content', '')
                        # Log the streaming chunk
                        logger.info(f"Streaming chunk from {node_name}: {chunk}")
                        # Yield the SSE chunk
                        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {'content': chunk}, 'finish_reason': None}]})}\n\n"
                except Exception as chunk_error:
                    # Log chunk-processing errors
                    logger.error(f"Error processing stream chunk: {chunk_error}")
                    continue

            # Yield the stream-termination marker
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        except Exception as stream_error:
            # Log stream-generation errors
            logger.error(f"Stream generation error: {stream_error}")
            # Yield an error payload
            yield f"data: {json.dumps({'error': 'Stream processing failed'})}\n\n"

    # Return the streaming response object
    return StreamingResponse(generate_stream(), media_type="text/event-stream")


# Dependency-injection helper for `graph` and `tool_config`
async def get_dependencies() -> Tuple[any, any]:
    """
    Dependency-injection helper that returns `graph` and `tool_config`.

    Returns:
        Tuple: A tuple containing `(graph, tool_config)`.

    Raises:
        HTTPException: Raised with status 500 if `graph` or `tool_config` is not initialized.
    """
    if not graph or not tool_config:
        raise HTTPException(status_code=500, detail="Service not initialized")
    return graph, tool_config

'''
request = {
        "messages": [{"role": "user", "content": user_message}],
        "stream": stream_flag,
        "userId": user_id,
        "conversationId": conversation_id
    }
'''
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, dependencies: Tuple[any, any] = Depends(get_dependencies)):
    """Receive frontend request data and process the business logic.

    Args:
        request: Request payload.

    Returns:
        Standard Python dictionary.
    """
    try:
        graph, tool_config = dependencies
        # Validate the request payload
        if not request.messages or not request.messages[-1].content:
            logger.error("Invalid request: Empty or invalid messages")
            raise HTTPException(status_code=400, detail="Messages cannot be empty or invalid")
        user_input = request.messages[-1].content
        logger.info(f"The user's user_input is: {user_input}")

        # Build runtime config with safe defaults for thread and user IDs
        config = {
            "configurable": {
                "thread_id": f"{getattr(request, 'userId', 'unknown')}@@{getattr(request, 'conversationId', 'default')}",
                "user_id": getattr(request, 'userId', 'unknown')
            }
        }

        # Handle streaming output
        if request.stream:
            return await handle_stream_response(user_input, graph, config)
        # Handle non-streaming output
        return await handle_non_stream_response(user_input, graph, tool_config, config)

    except Exception as e:
        logger.error(f"Error handling chat completion:\n\n {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    logger.info(f"Start the server on port {Config.PORT}")
    # `uvicorn` is a lightweight, high-performance ASGI server
    # It is used here to serve the asynchronous FastAPI application
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)


