# Shovs LLM OS Setup Guide

This guide covers the actual supported setup paths for the current codebase.

## What You Need

- Python 3.10+
- Node.js 18+
- npm 9+
- optional: Docker for SearXNG and sandbox services
- at least one model provider:
  - Ollama
  - LM Studio
  - llama.cpp
  - local OpenAI-compatible server
  - OpenAI / Groq / Anthropic / Gemini / Nvidia

## Install

### Backend

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Root Dependencies

```bash
npm install
```

### Frontend Dependencies

Pick the frontend you want to run:

```bash
cd frontend_nova && npm install && cd ..
```

```bash
cd frontend_consumer && npm install && cd ..
```

## Environment

Copy the example file:

```bash
cp .env.example .env
```

### Core Settings

```env
HOST=0.0.0.0
PORT=8000
DEBUG=True
DEFAULT_MODEL=llama3.2
```

### Provider Options

You can select one of the following:

```env
LLM_PROVIDER=auto
```

Supported values:
- `auto`
- `ollama`
- `lmstudio`
- `llamacpp`
- `local_openai`
- `openai`
- `groq`
- `anthropic`
- `gemini`
- `nvidia`

### Ollama

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.2
EMBED_MODEL=ollama:nomic-embed-text
```

The runtime supports both current Ollama embedding transport (`/api/embed`) and legacy (`/api/embeddings`) automatically.

### LM Studio

```env
LLM_PROVIDER=lmstudio
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_API_KEY=lm-studio
DEFAULT_MODEL=qwen2.5-coder-3b-instruct-mlx
EMBED_MODEL=lmstudio:text-embedding-nomic-embed-text-v1.5
```

### llama.cpp

```env
LLM_PROVIDER=llamacpp
LLAMACPP_BASE_URL=http://127.0.0.1:8080/v1
LLAMACPP_API_KEY=llama.cpp
```

### Local OpenAI-Compatible Server

```env
LLM_PROVIDER=local_openai
OPENAI_BASE_URL=http://127.0.0.1:9000/v1
OPENAI_API_KEY=local
EMBED_MODEL=text-embedding-3-small
```

### OpenAI Cloud

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

### Groq

```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

## Search Configuration

The web tooling can use:
- local SearXNG
- Brave
- Tavily
- Exa
- Groq-backed search flows

Example:

```env
SEARXNG_URL=http://localhost:8080
TAVILY_API_KEY=
BRAVE_SEARCH_KEY=
EXA_API_KEY=
```

## Optional Services

### SearXNG and Sandbox

```bash
npm run dev:services
```

This starts:
- `searxng`
- `agent-sandbox`

### MCP

MCP startup is disabled in local dev reload by default. The backend honors:

```env
ENABLE_MCP=true
ENABLE_MCP_IN_DEV=false
```

## Running the App

### Nova Workspace

```bash
npm run dev:nova
```

### Consumer Frontend

```bash
npm run dev:consumer
```

### Backend Only

```bash
npm run dev:backend
```

### Frontend Only

```bash
npm run dev:frontend:nova
```

### Consumer Only

```bash
npm run dev:frontend:consumer
```

## URLs

- Backend API: [http://localhost:8000](http://localhost:8000)
- FastAPI docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Nova frontend: check the Vite output port in terminal

## Recommended First Local Setup

### Option A: LM Studio on Apple Silicon

Good for:
- local-first development
- OpenAI-compatible transport
- small-model testing

Suggested model pattern:
- chat/planner/context: a small instruct model
- embeddings: a dedicated embedding model if available

### Option B: Ollama

Good for:
- simplest local setup
- fast smoke tests

### Option C: llama.cpp

Good for:
- lightweight local serving
- engineering-focused local workflows

## First Smoke Test

1. Start the backend.
2. Open Nova.
3. Select a model/provider.
4. Send:

```text
Research wigglebudget.com and give me a TLDR.
```

If you are using a small local model, start with:
- `Execution Loop = Single`
- `Manager Agent = off`
- `Max Tool Calls = 2 or 3`
- `Max Turns = 2 or 3`

Then move to `auto` or `managed` once the baseline is stable.

If you are testing semantic memory too, confirm `EMBED_MODEL` points at a reachable embedding model for the same provider family you are using.

## Troubleshooting

### Local model runs but the turn fails with context overflow

The runtime now retries local overflow cases with a smaller prompt, but you should still reduce pressure by:
- lowering `Max Tool Calls`
- lowering `Max Turns`
- using `Single` loop for small local models

### The model talks instead of calling a tool

This is usually a small-model tool-obedience issue, not a backend startup issue.

Recommended:
- try `Single` first
- reduce complexity of the request
- keep tool budgets low
- prefer exact-domain product research prompts over vague search tasks

### Memory storage fails with Ollama embedding errors

This usually means one of:
- the embed model is not available locally
- `EMBED_MODEL` still points at the wrong provider/model
- Ollama is up, but only the chat model is available

Recommended:
- make sure Ollama is running
- make sure the embedding model exists locally, for example `nomic-embed-text`
- set `EMBED_MODEL=ollama:nomic-embed-text` when using Ollama memory and retrieval
- restart the backend after changing provider or embedding settings

### Web search is unstable

This usually means:
- rate limits
- missing API keys
- network instability
- backend fallback path triggering

The runtime now degrades more aggressively instead of retrying forever, but external provider stability still matters.

## Storage

Important runtime stores:
- `sessions.db`
- `agents.db`
- `memory_graph.db`
- `tool_results.db`
- `runs.db`
- `chroma_db/`

Nova includes storage backup/reset controls for these stores.
