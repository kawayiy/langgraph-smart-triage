# LangGraph Smart Triage

An intelligent triage and health-record Q&A project built with `LangGraph`, `FastAPI`, `Gradio`, `Chroma`, and `PostgreSQL`.
The system is organized around the flow of "intent recognition -> tool calling -> retrieval grading -> query rewriting -> final generation", and supports streaming responses, multi-turn conversations, lightweight long-term memory, and health-record retrieval demos.

## Features

- Uses `LangGraph` to build a controllable agent workflow instead of a simple single-turn chatbot.
- Supports multiple model backends: `OpenAI`, `Qwen`, `OneAPI`, and `Ollama`.
- Includes a health-record retrieval tool `retrieve` and a calculation tool `multiply`.
- Grades retrieval relevance and rewrites the query automatically when results are not relevant enough.
- Provides three entry points: `FastAPI` API, `Gradio` web UI, and CLI interaction.
- Supports streaming output with an OpenAI-style Chat Completions interface.
- Uses `PostgreSQL` for LangGraph checkpoint/store persistence for conversation state and memory.

## Use Cases

- Health-record question answering
- Intelligent triage workflow demos
- A starter template for `RAG + Agent + LangGraph`
- Multi-model and multi-tool routing experiments

## Architecture

### 1. Interaction Layer

- `main.py`: FastAPI service exposing `POST /v1/chat/completions`
- `webUI.py`: Gradio frontend with login, registration, chat history, and conversation UI
- `ragAgent.py`: CLI debugging entry point
- `apiTest.py`: API integration test script

### 2. Agent Orchestration Layer

The LangGraph workflow is defined in `ragAgent.py`, with these core nodes:

- `agent`: analyzes the question, combines memory, and decides whether tools should be called
- `call_tools`: runs tools in parallel
- `grade_documents`: checks whether retrieved content is relevant
- `rewrite`: rewrites the query when retrieval quality is poor
- `generate`: produces the final answer

### 3. Tools and Knowledge Layer

- `utils/tools_config.py`: tool registration
- `vectorSave.py`: PDF chunking, embedding generation, and Chroma ingestion
- `utils/pdfSplitTest_Ch.py` / `utils/pdfSplitTest_En.py`: Chinese and English PDF splitting utilities
- `prompts/`: prompt templates for each node

### 4. Storage Layer

- `Chroma`: local vector store, default directory `chromaDB`
- `PostgreSQL`: used for LangGraph checkpoint and store
- `docker-compose.yml`: local PostgreSQL startup configuration

## Core Workflow

1. A user question enters the `agent` node.
2. The model decides whether it needs to call any tools.
3. If `retrieve` is called, the system first searches health records, then enters `grade_documents` to evaluate relevance.
4. If relevance is insufficient, the flow goes to `rewrite`, then returns to `agent` with the rewritten query.
5. If retrieval is good enough, or a non-retrieval tool was called, the flow goes to `generate`.
6. The final output is returned either as standard JSON or as an SSE stream.

## Project Structure

```text
langgraph-smart-triage/
├─ main.py                       # FastAPI service entry point
├─ ragAgent.py                   # LangGraph core logic and CLI entry
├─ webUI.py                      # Gradio demo frontend
├─ vectorSave.py                 # PDF -> vector store ingestion script
├─ apiTest.py                    # API test script
├─ docker-compose.yml            # PostgreSQL container config
├─ requirements.txt              # Python dependencies
├─ .env.example                  # Environment variable template
├─ prompts/
│  ├─ prompt_template_agent.txt
│  ├─ prompt_template_grade.txt
│  ├─ prompt_template_rewrite.txt
│  └─ prompt_template_generate.txt
└─ utils/
   ├─ config.py                  # Project configuration
   ├─ llms.py                    # Multi-model initialization
   ├─ tools_config.py            # Tool registration
   ├─ pdfSplitTest_Ch.py         # Chinese PDF splitting
   └─ pdfSplitTest_En.py         # English PDF splitting
```

## Requirements

- Python `3.10+`
- PostgreSQL `15` or a similar recent version
- One available LLM / embedding backend:
  - `Qwen / DashScope`
  - `OpenAI`
  - `OneAPI`
  - `Ollama`

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Environment Variables

First copy the template:

```bash
copy .env.example .env
```

Then fill in the required values:

```env
# OpenAI-compatible API
OPENAI_API_KEY=
OPENAI_BASE_URL=

# DashScope / Qwen
DASHSCOPE_API_KEY=

# OneAPI
ONEAPI_API_KEY=
ONEAPI_API_BASE=

# PostgreSQL
DB_URI=
POSTGRES_USER=
POSTGRES_PASSWORD=

# LangSmith (optional)
# LANGCHAIN_TRACING_V2=
# LANGCHAIN_API_KEY=
```

Notes:

- The default model type in `utils/config.py` is `qwen`
- The default backend port is `8013`
- `webUI.py` calls `http://localhost:8013/v1/chat/completions` by default

## Quick Start

### 1. Start PostgreSQL

If you use Docker:

```bash
docker compose up -d
```

Or:

```bash
docker-compose up -d
```

If you use a local PostgreSQL instance instead, make sure `DB_URI` is valid and the database is running.

### 2. Prepare the Knowledge Base

By default, the project uses `vectorSave.py` to split PDF content and write vectors into local Chroma storage.

Before running it, check these settings:

- `INPUT_PDF` in `vectorSave.py`
- `TEXT_LANGUAGE` in `vectorSave.py`
- `llmType` in `vectorSave.py`
- Whether the Chroma collection name in `vectorSave.py` matches the one in `utils/config.py`

Then run:

```bash
python vectorSave.py
```

By default, vectors are written into the `demo001` collection under the `chromaDB` directory.

### 3. Start the Backend API

```bash
python main.py
```

API endpoint:

`http://localhost:8013/v1/chat/completions`

### 4. Start the Web UI

```bash
python webUI.py
```

Default UI address:

`http://127.0.0.1:7861`

### 5. Run the CLI

```bash
python ragAgent.py
```

This is useful for quickly validating the main LangGraph workflow.

## API Usage

### Endpoint

`POST /v1/chat/completions`

### Request Example

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Query the health record of Zhang Sanjiu"
    }
  ],
  "stream": false,
  "userId": "8010",
  "conversationId": "8010"
}
```

### `curl` Example

```bash
curl -X POST "http://localhost:8013/v1/chat/completions" ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"Query the health record of Zhang Sanjiu\"}],\"stream\":false,\"userId\":\"8010\",\"conversationId\":\"8010\"}"
```

### Python Example

```python
import requests

url = "http://localhost:8013/v1/chat/completions"
payload = {
    "messages": [{"role": "user", "content": "Query the health record of Zhang Sanjiu"}],
    "stream": False,
    "userId": "8010",
    "conversationId": "8010",
}

resp = requests.post(url, json=payload, timeout=60)
print(resp.json())
```

### Streaming Responses

When `stream=true`, the backend returns `text/event-stream`, and each chunk follows an OpenAI-style chat completion chunk structure.

## Key Configuration

`utils/config.py` currently contains these important settings:

- `LLM_TYPE`: default model type, currently `qwen`
- `CHROMADB_DIRECTORY`: vector store directory, default `chromaDB`
- `CHROMADB_COLLECTION_NAME`: collection name, default `demo001`
- `DB_URI`: PostgreSQL connection string
- `HOST` / `PORT`: FastAPI host and port
- `LOG_FILE`: log file path, default `output/app.log`

## Prompt Templates

The `prompts/` directory contains the prompts used at each stage:

- `prompt_template_agent.txt`: triage logic and tool selection
- `prompt_template_grade.txt`: retrieval relevance grading
- `prompt_template_rewrite.txt`: query rewriting
- `prompt_template_generate.txt`: final response generation

## Implemented Capabilities

- RAG-based question answering over health records
- Simple calculator tool calling
- Multi-turn conversation
- Streaming and non-streaming API support
- Conversation-level state tracking
- User-level long-term memory
- Gradio demo frontend

## Current Limitations

- The user system in `webUI.py` is in-memory only, so accounts and conversation history are lost after restart.
- Vector store initialization depends on manually running `vectorSave.py`.
- The project is currently focused on the health-record QA demo, with only a small number of tools.
- Running the system requires external model services and database setup, so initial environment configuration is relatively heavy.

## Troubleshooting

### 1. Database connection error on startup

Check:

- Whether `DB_URI` is correct
- Whether PostgreSQL is running
- Whether username, password, and port match `docker-compose.yml`

### 2. No retrieval results

Check:

- Whether `python vectorSave.py` has been run
- Whether the collection name written by `vectorSave.py` matches `utils/config.py`
- Whether `INPUT_PDF` is correct
- Whether the embedding service is available

### 3. Web UI does not return results

Check:

- Whether `main.py` is running
- Whether the backend URL in `webUI.py` is still `http://localhost:8013/v1/chat/completions`
- Whether the correct port is being used

## Possible Next Steps

- Persist accounts and conversations from `webUI.py` into a database
- Add more medical assistant tools
- Add document upload and automatic vector ingestion
- Add API authentication and access control
- Add one-command containerized deployment

