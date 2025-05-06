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