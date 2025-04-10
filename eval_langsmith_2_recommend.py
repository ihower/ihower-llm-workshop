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

class Score(BaseModel):
  reasoning: str = Field(description="a explanation of your evaluation in Traditional Chinese (Taiwan)")
  score: int = Field(description="1~5")
  
def my_llm_call(inputs: dict) -> dict:
    answer = recommend_book(inputs["title"], inputs["description"])

    return { "answer": answer }

def my_llm_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    criteria = """答案必須
1. 簡潔度（25%）：推薦文案是否在15-30字內完成表達，簡短有力不冗長
2. 吸引力（25%）：是否使用吸引人的語言、修辭或問題引發讀者興趣，讓人想進一步了解
3. 相關性（25%）：是否準確反映書籍的核心主題、價值或獨特賣點
4. 目標性（25%）：是否清楚指出適合的讀者群體或能解決的問題"""

    question = "根據以下書籍資料，如果我想用一句話推薦給別人，應該要說什麼? \n" + str(inputs["title"]) + " \n" + str(inputs["description"])[0:50]
    answer = outputs["answer"]

    prompt = f"""
[Question]

{question}

[The Start of Assistant's Answer]

{answer}

[The End of Assistant's Answer]

[Criteria]

{criteria}

[System]

We would like to request your feedback on the performance of an AI assistant in response to the user question displayed above.
Please rate their response according to the criteria defined above. Do not consider any other criteria.
Rating scale:

1: Fails to meet most criteria, needs complete revision
2: Meets few criteria, requires significant modification
3: Partially meets criteria, needs moderate revision
4: Mostly meets criteria, needs minor adjustments
5: Completely meets all criteria, no changes needed

Please evaluate each criterion separately and then provide an overall score on a scale of 1 to 5."""

    completion = openai_client.beta.chat.completions.parse(
      model="gpt-4o",
      messages = [
        {
            "role": "user",
            "content": prompt
        }
      ],
      response_format=Score
    )  

    eval_data = completion.choices[0].message.parsed

    return (eval_data.score / 5)

experiment_results = client.evaluate(
    my_llm_call,
    data="Book Dataset 107",
    evaluators=[my_llm_evaluator],
    experiment_prefix="my-recommend-llm-eval",
    max_concurrency=10,
)