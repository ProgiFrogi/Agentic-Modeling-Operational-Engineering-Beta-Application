# Agentic-Modeling-Operational-Engineering-Beta-Application

Multi-agent system for Kaggle-style competitions (course **Agentic Systems**, MWS): **Planner**, **Data Analytic**, **Data Worker**, **Coder**, and **Performance Assessor**, orchestrated with **LangGraph**, **OpenAI** tool-calling, plus optional **RAG** over indexed Kaggle notebooks.

## Architecture

- **Tools** ([`tools/`](tools/)): safe CSV preview/inspect, Python validate/execute in a workspace, `retrieve_code` (Chroma + embeddings), Kaggle download/submit helpers ([`tools/kaggle_utils.py`](tools/kaggle_utils.py)).
- **Agents** ([`agents/`](agents/)): role prompts + ReAct-style tool loops ([`agents/tool_loop.py`](agents/tool_loop.py)).
- **Workflow** ([`workflows/kaggle_workflow.py`](workflows/kaggle_workflow.py)): LangGraph state machine — planner routes to specialists, optional submit + assessor feedback loop.
- **API** ([`api/main.py`](api/main.py)): FastAPI — `POST /competition/start`, `GET /competition/{run_id}/status`, `WebSocket /ws/competition/{run_id}` (snapshot on connect).
- **RAG** ([`rag/`](rag/)): existing pipeline; use `KaggleRAGPipeline.build_index_for_competition()` to focus indexing on one competition’s public kernels.

## Setup

1. Python 3.10+ and a virtualenv:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Copy [`.env.example`](.env.example) to `.env`. Either set **`OPENAI_API_KEY`** for cloud OpenAI, or switch to **local Ollama** (see below).

3. **Kaggle** — pick one:
   - Classic: `~/.kaggle/kaggle.json`, or  
   - **New token:** set `KAGGLE_API_TOKEN=KGAT_...` in `.env` (same as in the shell `export ...`).  
   The first Kaggle call in this project runs `load_dotenv()` then authenticates, so **`.env` is enough for scripts/agents** without exporting manually.  
   Do **not** commit tokens; rotate any token that was ever pasted into a chat or screenshot.

### Kaggle data download vs OpenAI quota

- **Downloading competition files** uses **only the Kaggle API** (your `kaggle.json`). It does **not** use OpenAI and does **not** consume any LLM quota.
- If you see **`403 Forbidden`** on download, you usually need to open the competition page on Kaggle, **join / accept rules**, then retry. It is an **access** issue on Kaggle’s side, not an AI limit.
- **`429` / `insufficient_quota` from OpenAI** happens only when **agents** (Planner, etc.) call the language model after the download step.

To **only download data** and exit (no agents, no LLM):

```bash
./run.sh --competition mws-ai-agents-2026 --download-only
```

(`run.sh` sets `PYTHONPATH` to the repo root; see below.)

### Why `PYTHONPATH=.` (and `run.sh`)

`.env` задаёт **переменные для приложения** (OpenAI, Kaggle token и т.д.), но **не** говорит Python, где искать пакеты `config`, `agents`, `workflows`, `rag`, `tools`.  
При запуске `python cli.py` из корня репозитория корень проекта **не всегда** в `sys.path`, поэтому без `PYTHONPATH=.` часто падает `ModuleNotFoundError: config`.

**Простой запуск** (из корня репо, после `source venv/bin/activate`):

```bash
./run.sh
```

В `.env` уже заданы **`COMPETITION_REF`** и **`WORKSPACE_ROOT`** (по умолчанию данные в `./workspace/<slug>/`). Дополнительные флаги нужны только чтобы **переопределить** это на один запуск.

- Подробные логи **включены по умолчанию** (поток графа в stderr + `INFO` для `agentic.*`). Выключить: **`WORKFLOW_VERBOSE=0`** в `.env` и/или **`./run.sh -q`**
- Другой конкурс / путь на раз: `./run.sh --competition other-comp --workspace ./workspace/other-comp`
- Явный каталог данных всегда один и тот же: задай **`WORKSPACE_PATH=./workspace/mws-ai-agents-2026`** в `.env`
- **Чистый прогон без старых скриптов:** `./run.sh --fork-workspace` создаёт папку `workspace/<slug>_<timestamp>_<pid>` с симлинками на `train.csv` / `test.csv` / sample из основного workspace (сначала должен быть скачан датасет в базовую папку).
- **Убрать мусор в том же workspace:** `./run.sh --clean-agent-artifacts` перед запуском удаляет `scripts/_agent_*.py`, `tmp*.py` в корне и все `__pycache__` под workspace (меньше конфликтов `import clean_data`).

Скрипт [`run.sh`](run.sh) сам выставит `PYTHONPATH` и возьмёт `python` из активного `venv` / `venv` / `.venv` при наличии.

Эквивалент вручную:

```bash
cd /path/to/Agentic-Modeling-Operational-Engineering-Beta-Application
source venv/bin/activate
export PYTHONPATH=.
python cli.py --competition mws-ai-agents-2026 --workspace ./workspace/mws-ai-agents-2026
```

### Kaggle: личный аккаунт и «команда»

Отдельного «API от имени команды» обычно **нет**: запросы идут с **твоего** токена / `kaggle.json`. На сайте Kaggle ты **создаёшь или вступаешь в Team** для соревнования; **сабмит от твоего аккаунта** (через этот же API) в рамках правил зачёта может **идти в зачёт команды**, если ты в составе команды и соревнование это допускает.  
Сокомандники **не делят один секретный токен** как обязательную схему: у каждого свой доступ; кто-то может сабмитить у себя локально от своего аккаунта — это нормальная модель. Детали (кто может сабмитить, лимиты) — в правилах конкретного соревнования на Kaggle.

### Local LLM with Ollama (no OpenAI billing)

1. Install [Ollama](https://ollama.com/) and start the app (daemon listens on port **11434**).
2. Pull a model (name must match `ollama list` exactly). Instruct-tuned models work better for tools/JSON than base models, e.g.:

```bash
ollama pull qwen2.5:7b
```

3. In `.env`, point the app at Ollama’s OpenAI-compatible API (see [`.env.example`](.env.example)). Use the **exact** model tag from `ollama list`:

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2.5:7b
OPENAI_PLANNER_MODEL=qwen2.5:7b
```

4. Run as usual: `./run.sh ...` (or `export PYTHONPATH=.` + `python cli.py ...`).

**Caveat:** local models are weaker at **structured outputs** and **tool calling** than GPT-4-class models; if the planner fails or loops, try a larger model or shorter `MAX_WORKFLOW_ITERATIONS`.

## Run agents (CLI)

Download competition data (train/test/sample submission) into the workspace, then run the graph:

```bash
./run.sh --competition mws-ai-agents-2026 --download-data
```

Or only run (data already present under `WORKSPACE_ROOT/<slug>/`):

```bash
./run.sh --competition mws-ai-agents-2026 --workspace ./workspace/mws-ai-agents-2026
```

Progress logs (stderr) after each graph step:

```bash
./run.sh --competition mws-ai-agents-2026 --workspace ./workspace/mws-ai-agents-2026
```

С **`WORKFLOW_VERBOSE=0`** или **`./run.sh -q`** поток шагов графа и agentic-логи почти отключаются (останется строка `Finished...` и предупреждения библиотек).

Tune planner passes via env: `MAX_WORKFLOW_ITERATIONS` (default 10) and `FORCE_CODER_MIN_PLANNER_ITERATION` (default 3 — after that, prep steps are skipped until `submission.csv` exists). See [`config/settings.py`](config/settings.py).

## Run API

```bash
# из корня репозитория:
export PYTHONPATH=.
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

- `POST /competition/start` JSON: `{"competition_ref": "mws-ai-agents-2026", "workspace_dir": null}`
- `GET /competition/{run_id}/status`

## Build RAG index (optional)

Indexing still uses an OpenAI-compatible endpoint for code descriptions (see [`rag/init.py`](rag/init.py)). For **one competition**:

```python
from openai import OpenAI
from rag.pipeline import KaggleRAGPipeline
from config.settings import get_settings

s = get_settings()
client = OpenAI()  # or Ollama-compatible base_url
pipe = KaggleRAGPipeline(
    client,
    s.openai_model,
    vector_store_path=s.rag_vector_store_path,
)
pipe.build_index_for_competition("mws-ai-agents-2026", notebooks_per_comp=8)
```

Agents call `tool_retrieve_code` which reads from `RAG_VECTOR_STORE_PATH`.

## Tests

```bash
pytest
```

## Notes

- **Execution safety**: `execute_code` runs a temp script under the workspace with a timeout; it is not a full sandbox — use a dedicated user/VM for untrusted competitions.
- **Costs**: workflow issues many LLM + tool calls; cap iterations and model sizes for experiments.
