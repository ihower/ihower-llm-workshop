#  more build-in evaluator:
#
#  https://github.com/langchain-ai/openevals for LangSmith
#  https://github.com/braintrustdata/autoevals for Braintrust

import os
from typing import List
from pydantic import Field, BaseModel, ConfigDict
from openai import OpenAI

from langsmith import Client
from my_app import recommend_book
client = Client()

openai_api_key = os.environ.get("OPENAI_API_KEY", "sk-proj-1234567890")

openai_client= OpenAI(api_key=openai_api_key)

def my_llm_call(inputs: dict) -> dict:
    answer = recommend_book(inputs["title"], inputs["description"])

    return { "answer": answer, "context": inputs }

#------

from openevals.llm import create_llm_as_judge
from openevals.prompts import CONCISENESS_PROMPT, HALLUCINATION_PROMPT

def my_llm_as_judge1(inputs: dict, outputs: dict, reference_outputs: dict):
    llm_as_judge1 = create_llm_as_judge(
      prompt=CONCISENESS_PROMPT,
      feedback_key="conciseness",
      model="openai:gpt-4.1-mini",
    )

    return llm_as_judge1(
        inputs="根據以下書籍資料，如果我想用一句話推薦給別人，應該要說什麼? \n" + str(inputs["title"]) + " \n" + str(inputs["description"])[0:50],
        outputs=outputs["answer"]
    )

def my_llm_as_judge2(inputs: dict, outputs: dict, reference_outputs: dict):
    llm_as_judge2 = create_llm_as_judge(
        prompt=HALLUCINATION_PROMPT,
        feedback_key="hallucination",
        model="openai:gpt-4.1-mini",
    )

    return llm_as_judge2(
        inputs="根據以下書籍資料，如果我想用一句話推薦給別人，應該要說什麼?",
        outputs=outputs["answer"],
        context=outputs["context"],
        reference_outputs=""
    )

experiment_results = client.evaluate(
    my_llm_call,
    data="Book Dataset 107",
    evaluators=[my_llm_as_judge1, my_llm_as_judge2],
    experiment_prefix="my-recommend-llm-openevals",
    max_concurrency=10,
)