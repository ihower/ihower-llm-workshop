import os
from typing import List
from pydantic import Field, BaseModel, ConfigDict
from openai import OpenAI
from braintrust import Eval, init_dataset, init_logger, traced, wrap_openai

from my_app import recommend_book

client= OpenAI(api_key=os.environ['OPENAI_API_KEY'])
openai_client = wrap_openai(client)

class Score(BaseModel):
  reasoning: str = Field(description="a explanation of your evaluation in Traditional Chinese (Taiwan)")
  score: int = Field(description="1~5")
  
def my_llm_call(input: dict) -> dict:
    answer = recommend_book(input["title"], input["description"])
    return { "answer": answer }

def score_with_gpt(input, expected, output):
    criteria = """答案必須
1. 簡潔度（25%）：推薦文案是否在15-30字內完成表達，簡短有力不冗長
2. 吸引力（25%）：是否使用吸引人的語言、修辭或問題引發讀者興趣，讓人想進一步了解
3. 相關性（25%）：是否準確反映書籍的核心主題、價值或獨特賣點
4. 目標性（25%）：是否清楚指出適合的讀者群體或能解決的問題"""

    question = "根據以下書籍資料，如果我想用一句話推薦給別人，應該要說什麼? \n" + str(input["title"]) + " \n" + str(input["description"])[0:50]
    answer = output["answer"]

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

Please evaluate each criterion separately and then provide an overall score on a scale of 1 to 5.
只需回傳一個整體分數（1~5），不需說明理由。"""

    completion = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    # 只取數字分數
    try:
        score = int(''.join(filter(str.isdigit, completion.choices[0].message.content.strip())))
        return score / 5
    except Exception:
        return 0

Eval(
    "Course-202504", # project name
    task=my_llm_call,
    data=init_dataset(project="Course-202504", name="books-107"),
    scores=[score_with_gpt],
    experiment_name="recommend-llm-eval"
)