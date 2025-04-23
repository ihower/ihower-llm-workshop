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

安裝 https://www.promptfoo.dev/

```
npx promptfoo@latest eval -c promptfooconfig.yaml
npx promptfoo@latest view
```