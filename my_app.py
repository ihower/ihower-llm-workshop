import requests
import json
from pprint import pp

import os
openai_api_key = os.environ.get("OPENAI_API_KEY", "sk-proj-1234567890")

def get_completion(messages, model="gpt-4o-mini", temperature=0, max_tokens=600, format_type=None):
  payload = { "model": model, "temperature": temperature, "messages": messages, "max_tokens": max_tokens }
  if format_type:
    payload["response_format"] =  { "type": format_type }

  headers = { "Authorization": f'Bearer {openai_api_key}', "Content-Type": "application/json" }
  response = requests.post('https://api.openai.com/v1/chat/completions', headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)
  if response.status_code == 200 :
    return obj["choices"][0]["message"]["content"]
  else :
    return obj["error"]
  
def classify_book(name, description):
  messages = [
    {
      "role": "system",
      "content": """You are tasked with categorizing a book based on its information. Your goal is to select the most appropriate skill category from a predefined list.
Here is the list of available categories to choose from:
<category>
程式語言, Data Science, 人工智慧, 分散式架構, 系統開發, 行動軟體開發, 資料庫, 資訊科學, 軟體架構, 軟體測試, 軟體工程, 資訊安全, 網站開發, 前端開發, 架站軟體, 網頁設計, Adobe 軟體應用, Office 系列, 遊戲開發設計, UI/UX, 雲端運算, 區塊鏈與金融科技, 物聯網 IoT, 商業管理類, 電子電路電機類, 嵌入式系統, 視覺影音設計, 考試認證, 數學, 微軟技術, MAC OS 蘋果電腦, 其他, 兒童專區, 製圖軟體應用, 語言學習, 國家考試, 職涯發展, Java, 理工類, 網路通訊, 量子電腦
</category>
Analyze the book information and determine which category best fits the content and subject matter of the book. Consider the main topics, technologies, or skills covered in the book when making your decision.
Choose only one category that most accurately represents the book's primary focus. Do not select multiple categories or create new ones.
Provide your answer by outputting only the chosen category name, without any additional text or explanation. Your response should consist solely of the category name.
Remember to choose the most appropriate category based on the book information provided.""" 
    },
    {
      "role": "user",
      "content": f"書名: {name} \n 描述: {description}"
    }
  ]
  response = get_completion(messages, model="gpt-4o-mini")
  return response


def recommend_book(name, description, model="gpt-4o-mini"):
  messages = [
    {
      "role": "system",
      "content": "根據以下書籍資料，如果我想用一句話推薦給別人，應該要說什麼?" 
    },
    {
      "role": "user",
      "content": f"書名: {name} \n 描述: {description}"
    }
  ]
  response = get_completion(messages, model=model, temperature=0.5)
  return response