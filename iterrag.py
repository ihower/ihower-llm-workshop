# 基於 https://arxiv.org/abs/2410.04343v1 的 IterRAG 實作

from typing import List, Dict, Any, Callable
from pydantic import BaseModel, Field
from openai import OpenAI
from tavily import TavilyClient
import os

from dotenv import load_dotenv
load_dotenv(".env", override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY") 

openai_client = OpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

DEFAULT_MODEL = "gpt-4o"

class QueryResult(BaseModel):
    type: str = Field(description="Response type: either 'sub_query', 'intermediate_answer', or 'final_answer'")
    content: str = Field(description="The response content")

def perform_tavily_search(query):
  search_response = tavily_client.search(query)
  contexts = [ x["content"] for x in search_response["results"] ]
  return contexts

def format_prompt(query: str, documents: List[str], conversation_history: List[Dict] = None) -> str:
    # 添加當前問題及上下文
    prompt = "## Context:\n"
    for i, doc in enumerate(documents):
        prompt += f"<chunk>{doc}</chunk>\n"
    
    prompt += f"\nUser Question: {query}\n\n"

    # 如果有對話歷史，添加到提示詞中
    if conversation_history and len(conversation_history) > 0:
        prompt += "## Conversation History:\n"
        for turn in conversation_history:
            if "sub_query" in turn:
                prompt += f"Sub Question: {turn['sub_query']}\n"
            if "intermediate_answer" in turn:
                prompt += f"Intermediate Answer: {turn['intermediate_answer']}\n"

    return prompt

def generate_llm_response(prompt: str) -> str:
    messages = [
            {"role": "system", "content": """You are an expert question-answering assistant 
             with the ability to decompose complex questions into manageable sub-questions.

When responding to a question, follow this systematic approach:

1. Analyze the question to determine if it requires decomposition
2. If needed, break down the complex question into logical sub-questions          
3. Before generating any sub-query, check the Conversation History to avoid duplicating previous sub-queries             
4. Process each sub-question according to these rules:
   - When you can directly answer a sub-question, respond with type 'intermediate_answer'
   - When you need additional information, generate a relevant sub-question with type 'sub_query'
   - After generating a sub-query, proceed to answer it in your next response
   - If you notice that a sub-query has already been generated but not answered, do not regenerate it - instead, provide your best possible answer based on available information
   - When you have gathered sufficient information to address the original question, provide a comprehensive solution with type 'final_answer'

All responses must follow the specified json format:
- type: Either 'sub_query', 'intermediate_answer', or 'final_answer'
- content: Your response text

Important notes for final_answer:
- The final_answer must be in Taiwanese Chinese
- The final_answer should be comprehensive and detailed
- Provide a thorough explanation with sufficient length proportional to the context information available
- Synthesize all intermediate findings into a cohesive, well-structured response"""},
            {"role": "user", "content": prompt}
        ]
    
    print(f"\n>>>>> AI request: \n\n{messages}\n")
    response = openai_client.beta.chat.completions.parse(
        model=DEFAULT_MODEL,
        messages=messages,
        temperature=0,
        response_format=QueryResult
    )

    parsed_response = response.choices[0].message.parsed
    print(f"\n<<<<< AI response: \n\n {parsed_response}\n")
    return parsed_response


def iterative_rag(query: str, max_iterations: int, num_documents: int) -> Dict:
    # 初始檢索
    documents = perform_tavily_search(query)
    
    # 記錄對話歷史
    conversation_history = []
    
    # 初始化結果
    result = {
        "query": query,
        "documents": documents.copy(),
        "conversation": [],
        "final_answer": None
    }
    
    # 迭代直到找到最終答案或達到最大迭代數
    iteration = 0
    while iteration < max_iterations:
        print(f"\n=== 第 {iteration + 1} 次迭代 ===")
        iteration += 1

        # 準備提示詞
        prompt = format_prompt(query, documents, conversation_history)
        
        # 向LLM發送請求
        parsed_response = generate_llm_response(prompt)
        
        # 記錄對話歷史
        if parsed_response.type == "sub_query":
            # 添加子問題到歷史
            sub_query = parsed_response.content
            conversation_history.append({"sub_query": sub_query})
            result["conversation"].append({"sub_query": sub_query})
            
            # 檢索子問題的文檔
            sub_documents = perform_tavily_search(sub_query)[:num_documents]
            documents.extend(sub_documents)
            documents = list(set(documents))
            result["documents"] = documents
            
        elif parsed_response.type == "intermediate_answer":
            # 添加中間答案到歷史
            intermediate_answer = parsed_response.content
            conversation_history.append({"intermediate_answer": intermediate_answer})
            result["conversation"].append({"intermediate_answer": intermediate_answer})
            
        elif parsed_response.type == "final_answer":
            # 獲取最終答案
            final_answer = parsed_response.content
            result["final_answer"] = final_answer
            result["conversation"].append({"final_answer": final_answer})
            break
    
    # 如果達到最大迭代數但沒有最終答案，強制生成一個
    if not result["final_answer"] and iteration >= max_iterations:
        final_prompt = format_prompt(query, documents, conversation_history)

        messages = [
            {"role": "system", "content": f"""You are an expert question-answering assistant. Important notes for final_answer:
- The final_answer must be in Taiwanese Chinese             
- The final_answer should be comprehensive and detailed
- Provide a thorough explanation with sufficient length proportional to the context information available
- Synthesize all intermediate findings into a cohesive, well-structured response"""},
            {"role": "user", "content": final_prompt},
            {"role": "system", "content": "Based on the context and conversation history, provide the final answer in Taiwanese Chinese. "}
        ]
    
        response = openai_client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=messages,
            temperature=0
        )
        
        print(f"\n>>>>> 超過迭代次數，需要提前結束 AI request: \n\n{messages}\n")

        response = response.choices[0].message.content
        
        result["final_answer"] = response
        result["conversation"].append({"final_answer": response}) 
            
    return result

def main():
    # 創建 OpenAI 客戶端
    client = OpenAI()
    
    # 使用示例查詢
    query = "台灣最具戰略價值的供應鏈是什麼? 這個供應鏈在中國的表現如何?" 
    #query = "愛因斯坦和瑪麗·居里，誰的壽命更長？"

    result = iterative_rag(
        query=query,
        max_iterations=10,
        num_documents=5
    )
    
    # 輸出結果
    #print("\n----------- 中間過程:\n")
    #print(result)
    print("\n\n---- 最終答案:\n")
    print(result["final_answer"])

if __name__ == "__main__":
    main()