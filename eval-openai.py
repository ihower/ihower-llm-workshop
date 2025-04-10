# npx promptfoo@latest eval -c promptfooconfig.yaml
# npx promptfoo@latest view

import sys
import os
import json
from my_app import recommend_book

def call_api(prompt, options, context):
    config = options.get('config', None)    
    llm_model = config["model"]

    title = context['vars'].get('title', None)
    description = context['vars'].get('description', None)

    answer = recommend_book(title, description, llm_model)

    result = {
        "output": answer,
        "metadata": {
            "title": title
        },
    }

    return result
