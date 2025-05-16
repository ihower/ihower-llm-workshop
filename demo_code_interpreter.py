import os

from dotenv import load_dotenv
load_dotenv(".env", override=True)

from python_code_interpreter_tool import PythonExecTool
from openai import OpenAI
import requests
import json

openai_api_key = os.getenv("OPENAI_API_KEY")

def get_completion(messages, model="gpt-4.1-mini", temperature=0, tools=None, tool_choice=None, parallel_tool_calls=True):
  payload = { "model": model, "store": True, "temperature": temperature, "messages": messages, "store": True }
  if tools:
    payload["tools"] = tools
    payload["parallel_tool_calls"] = parallel_tool_calls

    if tool_choice:
      payload["tool_choice"] = tool_choice

  headers = { "Authorization": f'Bearer {openai_api_key}', "Content-Type": "application/json" }
  response = requests.post('https://api.openai.com/v1/chat/completions', headers = headers, data = json.dumps(payload) )
  obj = json.loads(response.text)

  if response.status_code == 200 :
    return obj["choices"][0]["message"] # 改成回傳上一層 message 物件
  else :
    return obj["error"]
  
def execute_python_code(python_code: str) -> str:
    """
    Execute the provided Python code and return the output.
    """
    tool = PythonExecTool()
    try:
        # 將字串包裝成字典格式
        result = tool.run({"python_code": python_code})
        return str(result)
    except Exception as e:
        import traceback
        return f"執行時發生錯誤: {str(e)}\n{traceback.format_exc()}"

available_tools = {
  "execute_python_code": execute_python_code,
}

def run_full_turn(messages, model="gpt-4.1", temperature=0, tools=None, tool_choice=None, parallel_tool_calls=True):
  # print(f"💬 呼叫 API: {messages}")
  response = get_completion(messages, model=model, temperature=temperature, tools=tools,tool_choice=tool_choice,parallel_tool_calls=parallel_tool_calls)

  if response.get("tool_calls"): # 或用 response 裡面的 finish_reason 判斷也行
    messages.append(response)

    # ------ 呼叫函數，這裡改成執行多 tool_calls (可以改成平行處理，目前用簡單的迴圈)
    for tool_call in response["tool_calls"]:
      function_name = tool_call["function"]["name"]
      function_args = json.loads(tool_call["function"]["arguments"])
      function_to_call = available_tools[function_name]

      print(f"   ⚙️ 呼叫函式 {function_name} 參數 {function_args}")
      function_response = function_to_call(**function_args)
      messages.append(
          {
              "tool_call_id": tool_call["id"], # 多了 toll_call_id
              "role": "tool",
              "name": function_name,
              "content": str(function_response), # 一定要是字串
          }
      )

    # 進行遞迴呼叫
    return run_full_turn(messages, model=model, temperature=temperature, tools=tools,tool_choice=tool_choice,parallel_tool_calls=parallel_tool_calls)

  else:
    return response["content"] 


system_prompt = """You are a helpful data science assistant. Your role is to analyze data and generate Python code to answer user queries effectively.

## Core Responsibilities

1. Process data as specified by the user
2. Generate Python code for data analysis
3. Execute the code using the `execute_python_code` tool
4. Explain the findings to the user

## Technical Capabilities

You can utilize the following Python libraries:
- pandas
- numpy
- matplotlib and seaborn
- scikit-learn

## Workflow

1. Understand the user's question and data
2. Write efficient Python code
3. Run the code using the tool
4. Interpret the results and provide insights

## Output Requirements

Always provide your explanations and insights in Traditional Chinese (Taiwan). The code will be in Python (English), but all your analysis, interpretations, and communications must be in Chinese.
"""

messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": """我想了解Python數據分析的基本流程。請示範如何：

1. 生成一個模擬的銷售數據集，包含日期、產品類別、銷售額和客戶年齡
2. 計算並顯示各產品類別的銷售總額和平均值
3. 計算銷售額與客戶年齡的相關係數
4. 使用簡單的機器學習模型預測銷售額，並顯示預測準確度
5. 輸出關鍵統計指標（如最高/最低銷售額、標準差等）

不需要使用任何外部文件，請直接在代碼中生成模擬數據，並只使用 print 輸出計算結果，不需要視覺化圖表。"""}]
tools = [PythonExecTool().get_definition()]

response = run_full_turn(messages, tools=tools)
print("------")
print(response)
