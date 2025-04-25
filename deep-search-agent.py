import os
import json
import aiohttp
from pydantic import BaseModel, Field

from agents import Agent, Runner, function_tool

from tavily import AsyncTavilyClient

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
JINA_API_KEY = os.environ.get("JINA_API_KEY")
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)

@function_tool
async def perform_web_search(query: str):
    """Perform web search"""
    print(f"  Performing Tavily search for {query}")

    search_response = await tavily_client.search(query, max_results=3)
    contexts = search_response["results"]
    # print(f"    Tavily search results: {contexts}")
    return str(contexts)

@function_tool
async def fetch_webpage_text(url: str):
    """Fetch webpage content"""

    print(f"  Fetching webpage text from {url}")
    full_url = f"https://r.jina.ai/{url}"
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}"
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(full_url, headers=headers) as resp:
                if resp.status == 200:
                    return await resp.text()
                else:
                    text = await resp.text()
                    print(f"Jina fetch error for {url}: {resp.status} - {text}")
                    return ""
    except Exception as e:
        print(f"從 Jina 獲取網頁內容時發生錯誤：{e}")
        return "fetch error, please ignore this reference"

@function_tool
async def generate_intermediate_answer(answer: str):
    """Generate intermediate answer"""
    print(f"  生成中間答案：{answer}")
    return "ok"

@function_tool
async def generate_sub_questions(sub_questions: str):
    """Generate sub questions """
    print(f"  生成子問題：{sub_questions}")
    return "ok"


class SearchResult(BaseModel):
    references: list[str] = Field(
        description="URLs of the most relevant and authoritative sources used to formulate the answer"
    )
    final_answer: str = Field(
        description="Comprehensive response addressing the user's original question based on research findings"
    )
    
async def run_deep_search(user_query: str):
    print(f"  用戶問題: {user_query}")
    
    system_prompt = """You are an agent tasked to thoroughly research user questions by following this specific sequence:

## Required Process Flow

1. First use generate_sub_questions to break down the user's question into sub-questions
2. For EACH sub-question, complete this full cycle before moving to the next sub-question:
   - Use perform_web_search to find information specific to the current sub-question
   - Use fetch_webpage_text (multiple times as needed) to gather detailed information
   - Use generate_intermediate_answer to create a response for that specific sub-question
3. Only after ALL sub-questions have been individually researched and answered, provide a final comprehensive answer to the user's original query in Traditional Chinese (Taiwan)

## Tool Sequence Example

- generate_sub_questions
- perform_web_search (for sub-question 1)
- fetch_webpage_text (relevant page 1)
- fetch_webpage_text (relevant page 2)
- fetch_webpage_text (relevant page 3)
- generate_intermediate_answer (for sub-question 1)
- perform_web_search (for sub-question 2)
- fetch_webpage_text (relevant page 1)
- fetch_webpage_text (relevant page 2)
- fetch_webpage_text (relevant page 3)
- generate_intermediate_answer (for sub-question 2)
- [Continue this pattern for all sub-questions]

## Important Guidelines

- Process ONE sub-question completely before moving to the next
- Please keep going until the user's question is completely resolved before ending your turn
- Only terminate your turn when you are sure that ALL sub-questions have been thoroughly researched and answered
- The final answer to the user MUST be in Traditional Chinese (Taiwan)
"""

    agent = Agent( name="Deep Search", 
                   instructions=system_prompt,
                   model="gpt-4.1-2025-04-14", 
                   tools=[generate_sub_questions, perform_web_search, fetch_webpage_text, generate_intermediate_answer],
                   output_type=SearchResult
                 )

    result = await Runner.run(agent, input=user_query, max_turns=30) # Default is 10
    
    return result.final_output


if __name__ == "__main__":
    import sys
    import asyncio

    if len(sys.argv) < 2:
        print("用法: python deep-search-agent.py <查詢問題>")
        print("例如: python deep-search-agent.py \"什麼是 AGI?\"")
        sys.exit(1)
        
    user_query = " ".join(sys.argv[1:])

    result = asyncio.run(run_deep_search(user_query))

    print("------- 主要參考網址 -------")
    print(result.references)

    print("------- 最終回答 -------")
    print(result.final_answer)
