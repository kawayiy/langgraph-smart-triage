# LangGraph Smart Triage

一个基于 `LangGraph + FastAPI + Gradio + Chroma + PostgreSQL` 的智能分诊 / 健康档案问答项目。  
系统围绕“意图识别 -> 工具调用 -> 检索评估 -> 查询重写 -> 最终生成”构建，支持流式接口、多轮会话、简单长期记忆，以及健康档案检索场景演示。

## 项目特点

- 基于 `LangGraph` 构建可控的智能体流程，而不是单轮问答。
- 支持 `OpenAI`、`Qwen`、`OneAPI`、`Ollama` 多种模型接入方式。
- 提供健康档案检索工具 `retrieve` 和计算工具 `multiply`。
- 检索后会进行相关性打分，不相关时自动重写问题再检索。
- 提供三种使用方式：`FastAPI` 接口、`Gradio Web UI`、命令行交互。
- 支持流式输出，接口风格兼容 OpenAI Chat Completions。
- 使用 `PostgreSQL` 保存 LangGraph checkpoint / store，用于会话状态与记忆能力。

## 适用场景

- 健康档案问答
- 智能分诊流程演示
- RAG + Agent + LangGraph 项目模板
- 多模型、多工具路由实验

## 整体架构

### 1. 交互层

- `main.py`：FastAPI 服务，对外暴露 `POST /v1/chat/completions`
- `webUI.py`：Gradio 前端，支持登录、注册、历史会话、聊天
- `ragAgent.py`：命令行调试入口
- `apiTest.py`：接口联调脚本

### 2. 智能体编排层

`ragAgent.py`` 中定义了 LangGraph 流程，核心节点包括：

- `agent`：分析问题、结合记忆、决定是否调用工具
- `call_tools`：并行执行工具
- `grade_documents`：判断检索内容是否相关
- `rewrite`：在检索不佳时重写问题
- `generate`：生成最终回答

### 3. 工具与知识层

- `utils/tools_config.py`：工具注册
- `vectorSave.py`：PDF 切分、向量化、写入 Chroma
- `utils/pdfSplitTest_Ch.py` / `utils/pdfSplitTest_En.py`：中英文 PDF 切分处理
- `prompts/`：各节点提示词模板

### 4. 存储层

- `Chroma`：本地向量库，默认目录为 `chromaDB`
- `PostgreSQL`：用于 LangGraph checkpoint 与 store
- `docker-compose.yml`：提供本地 PostgreSQL 启动方式

## 核心流程

1. 用户提问进入 `agent` 节点。
2. 模型判断是否调用工具。
3. 若调用 `retrieve`，先检索健康档案，再进入 `grade_documents` 进行相关性判断。
4. 如果相关性不足，则进入 `rewrite`，重写问题后再回到 `agent`。
5. 如果检索结果有效，或调用的是普通工具，则进入 `generate` 生成最终答复。
6. 最终结果可通过普通 JSON 或 SSE 流返回。

## 项目结构

```text
langgraph-smart-triage/
├─ main.py                       # FastAPI 服务入口
├─ ragAgent.py                   # LangGraph 核心逻辑与 CLI 入口
├─ webUI.py                      # Gradio 演示前端
├─ vectorSave.py                 # PDF -> 向量库灌库脚本
├─ apiTest.py                    # API 调试脚本
├─ docker-compose.yml            # PostgreSQL 容器配置
├─ requirements.txt              # Python 依赖
├─ .env.example                  # 环境变量模板
├─ prompts/
│  ├─ prompt_template_agent.txt
│  ├─ prompt_template_grade.txt
│  ├─ prompt_template_rewrite.txt
│  └─ prompt_template_generate.txt
└─ utils/
   ├─ config.py                  # 项目配置
   ├─ llms.py                    # 多模型初始化
   ├─ tools_config.py            # 工具注册
   ├─ pdfSplitTest_Ch.py         # 中文 PDF 切分
   └─ pdfSplitTest_En.py         # 英文 PDF 切分
```

## 环境要求

- Python `3.10+` 或更高版本
- PostgreSQL `15` 左右版本
- 可用的 LLM / Embedding 服务之一：
  - `Qwen / DashScope`
  - `OpenAI`
  - `OneAPI`
  - `Ollama`

## 安装依赖

```bash
pip install -r requirements.txt
```

## 环境变量配置

先复制模板：

```bash
copy .env.example .env
```

按需填写以下配置：

```env
# OpenAI 兼容接口
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

# LangSmith（可选）
# LANGCHAIN_TRACING_V2=
# LANGCHAIN_API_KEY=
```

说明：

- 默认模型类型在 `utils/config.py` 中为 `qwen`
- 默认服务端口为 `8013`
- `webUI.py` 默认访问后端地址 `http://localhost:8013/v1/chat/completions`

## 快速开始

### 1. 启动 PostgreSQL

如果你使用 Docker：

```bash
docker compose up -d
```

或：

```bash
docker-compose up -d
```

如果使用本地 PostgreSQL，请确保 `DB_URI` 可连接，并且数据库已正常启动。

### 2. 准备知识库数据

项目默认通过 `vectorSave.py` 将 PDF 内容切分并写入本地 Chroma。

运行前请先检查这些配置：

- `vectorSave.py` 中的 `INPUT_PDF`
- `vectorSave.py` 中的 `TEXT_LANGUAGE`
- `vectorSave.py` 中的 `llmType`
- `utils/config.py` 与 `vectorSave.py` 中的 Chroma 集合名是否一致

然后执行：

```bash
python vectorSave.py
```

默认会将向量写入 `chromaDB` 目录下的 `demo001` 集合。

### 3. 启动后端 API

```bash
python main.py
```

启动后接口地址为：

`http://localhost:8013/v1/chat/completions`

### 4. 启动 Web UI

```bash
python webUI.py
```

默认访问地址：

`http://127.0.0.1:7861`

### 5. 命令行调试

```bash
python ragAgent.py
```

适合快速验证 LangGraph 主流程是否可用。

## API 用法

### 请求地址

`POST /v1/chat/completions`

### 请求体示例

```json
{
  "messages": [
    {
      "role": "user",
      "content": "查询张三九的健康档案信息"
    }
  ],
  "stream": false,
  "userId": "8010",
  "conversationId": "8010"
}
```

### `curl` 示例

```bash
curl -X POST "http://localhost:8013/v1/chat/completions" ^
  -H "Content-Type: application/json" ^
  -d "{\"messages\":[{\"role\":\"user\",\"content\":\"查询张三九的健康档案信息\"}],\"stream\":false,\"userId\":\"8010\",\"conversationId\":\"8010\"}"
```

### Python 示例

```python
import requests

url = "http://localhost:8013/v1/chat/completions"
payload = {
    "messages": [{"role": "user", "content": "查询张三九的健康档案信息"}],
    "stream": False,
    "userId": "8010",
    "conversationId": "8010",
}

resp = requests.post(url, json=payload, timeout=60)
print(resp.json())
```

### 流式返回

当 `stream=true` 时，后端会返回 `text/event-stream`，每个 chunk 的结构与 OpenAI 风格的 chat completion chunk 类似。

## 主要配置项

`utils/config.py` 中当前包含以下关键配置：

- `LLM_TYPE`：默认模型类型，当前为 `qwen`
- `CHROMADB_DIRECTORY`：向量库存储目录，默认 `chromaDB`
- `CHROMADB_COLLECTION_NAME`：集合名，默认 `demo001`
- `DB_URI`：PostgreSQL 连接串
- `HOST` / `PORT`：FastAPI 服务监听地址与端口
- `LOG_FILE`：日志文件，默认 `output/app.log`

## 提示词模板

`prompts/` 目录中包含智能体各阶段的提示词：

- `prompt_template_agent.txt`：工具选择与分诊
- `prompt_template_grade.txt`：检索结果相关性评分
- `prompt_template_rewrite.txt`：问题重写
- `prompt_template_generate.txt`：最终回答生成

## 当前已实现的能力

- 基于健康档案的 RAG 问答
- 简单计算工具调用
- 多轮对话
- 流式 / 非流式接口
- 会话级状态跟踪
- 用户级长期记忆
- Gradio 前端演示

## 当前限制

- `webUI.py` 中的用户系统是内存态，重启后不会保留账号和历史会话。
- 向量库构建依赖手动执行 `vectorSave.py`，不是自动初始化。
- 项目目前主要围绕健康档案问答演示构建，工具种类还比较少。
- 运行依赖外部模型服务与数据库环境，首次配置成本相对较高。

## 常见排查

### 1. 启动时报数据库连接错误

检查：

- `DB_URI` 是否正确
- PostgreSQL 是否已启动
- 用户名、密码、端口是否与 `docker-compose.yml` 一致

### 2. 检索不到内容

检查：

- 是否已执行 `python vectorSave.py`
- `vectorSave.py` 写入的集合名是否与 `utils/config.py` 中一致
- `INPUT_PDF` 是否正确
- embedding 服务是否可用

### 3. Web UI 无法返回结果

检查：

- `main.py` 是否已启动
- `webUI.py` 中的后端地址是否仍为 `http://localhost:8013/v1/chat/completions`
- 是否使用了正确的端口

## 后续可扩展方向

- 将 `webUI.py` 的账号与会话改为数据库持久化
- 增加更多医疗辅助工具
- 增加文档上传与自动灌库能力
- 增加接口鉴权与权限控制
- 增加容器化一键部署


