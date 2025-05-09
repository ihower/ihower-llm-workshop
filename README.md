## Setup

請先安裝 uv https://github.com/astral-sh/uv

```
uv sync

source .venv/bin/activate
```

## Prompt Engineering Example 1

```
python eval_langsmith_1_dataset.py
python eval_langsmith_2_category.py
python eval_langsmith_2_recommend.py
python eval_langsmith_2_recommend_openevals.py
```

## Prompt Engineering Example 2

```
python eval_braintrust_1_dataset.py
python eval_braintrust_2_category.py
python eval_braintrust_2_recommend.py
```

## Prompt Engineering Example 3

安裝 https://www.promptfoo.dev/

```
npx promptfoo@latest eval -c promptfooconfig.yaml
npx promptfoo@latest view
```

## FastAPI SSE Example

Copy `.env.example` to `.env` and edit it

```
uvicorn main:app --reload
```

Open

```
http://localhost:8000/static/completion.html
http://localhost:8000/static/structured_output.html
http://localhost:8000/static/agent.html
```


## Build Your Own Code Interpreter

參考自 https://cookbook.openai.com/examples/object_oriented_agentic_approach/secure_code_interpreter_tool_for_llm_agents

`docker build -t python_sandbox:latest .`

`docker run -d --name sandbox --network none --cap-drop all --pids-limit 64 --tmpfs /tmp:rw,size=64M   python_sandbox:latest sleep infinity`

`python demo_code_interpreter.py`

## MCP demo

`uv add "mcp[cli]"`

`mcp install mcp install my_mcp_server.py` 這會安裝到 claude app 中
`mcp dev my_mcp_server.py` 這會打開 MCP Inspector 在 http://127.0.0.1:6274