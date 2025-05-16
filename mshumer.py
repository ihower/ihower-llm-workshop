# 基於 https://github.com/mshumer/OpenDeepResearcher 修改而來
# 改用 OpenAI 和 Tavily search

import asyncio
import aiohttp
import json

from typing import List
from pydantic import Field, BaseModel

from openai import AsyncOpenAI
from tavily import TavilyClient

import os
from dotenv import load_dotenv
load_dotenv(".env", override=True)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY") 
JINA_API_KEY = os.environ.get("JINA_API_KEY")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

JINA_BASE_URL = "https://r.jina.ai/"
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_REPORT_MODEL = 'gpt-4o' # 最終生成報告用的模型

class SearchQueryResult(BaseModel):
    queries: List[str]

async def generate_first_search_queries(user_query):
    print("產生 search queries...")
    prompt = (
        "You are an expert research assistant. Given the user's query, generate up to four distinct, "
        "precise search queries that would help gather comprehensive information on the topic. "
        "Return only a list of strings, for example: ['query1', 'query2', 'query3']."
    )
    messages = [
        {"role": "system", "content": "You are a helpful and precise research assistant."},
        {"role": "user", "content": f"User Query: {user_query}\n\n{prompt}"}
    ]

    completion = await openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=SearchQueryResult
    )

    parsed_result = completion.choices[0].message.parsed
    return parsed_result.queries


async def perform_tavily_search(query, n=10):
    search_response = tavily_client.search(query)
    if search_response and "results" in search_response and len(search_response["results"]) > 0:
        results = search_response["results"][:n]
        links = [result.get("url") for result in search_response["results"] if "url" in result]
        return links
    else:
        print("在 Tavily 回應中沒有找到結果。")
        return []

async def fetch_webpage_text(url):
    full_url = f"{JINA_BASE_URL}{url}"
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
        return ""

class PageUsefulResult(BaseModel):
    answer: bool

async def is_page_useful(user_query, page_text):
    prompt = (
        "You are a critical research evaluator. Given the user's query and the content of a webpage, "
        "determine if the webpage contains information relevant and useful for addressing the query. "
        "Respond with exactly one word: 'Yes' if the page is useful, or 'No' if it is not. Do not include any extra text."
    )
    messages = [
        {"role": "system", "content": "You are a strict and concise evaluator of research relevance."},
        {"role": "user", "content": f"User Query: {user_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]

    completion = await openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=PageUsefulResult
    )

    parsed_result = completion.choices[0].message.parsed
    return parsed_result.answer

async def extract_relevant_context(user_query, search_query, page_text):
    prompt = (
        "You are an expert information extractor. Given the user's query, the search query that led to this page, "
        "and the webpage content, extract all pieces of information that are relevant to answering the user's query. "
        "Return only the relevant context as plain text without commentary."
    )
    messages = [
        {"role": "system", "content": "You are an expert in extracting and summarizing relevant information."},
        {"role": "user", "content": f"User Query: {user_query}\nSearch Query: {search_query}\n\nWebpage Content (first 20000 characters):\n{page_text[:20000]}\n\n{prompt}"}
    ]

    response = await openai_client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=messages
    )
    return response.choices[0].message.content.strip()

async def get_new_search_queries(user_query, previous_search_queries, all_contexts):
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an analytical research assistant. Based on the original query, the search queries performed so far, "
        "and the extracted contexts from webpages, determine if further research is needed. "
        "If further research is needed, provide up to four new search queries as a Python list (for example, "
        "['new query1', 'new query2']). If you believe no further research is needed, respond with exactly [] empty list."
    )
    messages = [
        {"role": "system", "content": "You are a systematic research planner."},
        {"role": "user", "content": f"User Query: {user_query}\nPrevious Search Queries: {previous_search_queries}\n\nExtracted Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]

    completion = await openai_client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=messages,
        response_format=SearchQueryResult
    )

    parsed_result = completion.choices[0].message.parsed
    return parsed_result.queries

async def generate_final_report(user_query, all_contexts):
    context_combined = "\n".join(all_contexts)
    prompt = (
        "You are an expert researcher and report writer. Based on the gathered contexts below and the original query, "
        "write a comprehensive, well-structured, and detailed report that addresses the query thoroughly. "
        "Include all relevant insights and conclusions without extraneous commentary."
        "輸出請用台灣繁體中文"
    )
    messages = [
        {"role": "system", "content": "You are a skilled report writer."},
        {"role": "user", "content": f"User Query: {user_query}\n\nGathered Relevant Contexts:\n{context_combined}\n\n{prompt}"}
    ]

    print("--------------------------------")
    print(f"最終報告生成 prompt (共有 {len(all_contexts)} 筆參考資料): ", messages)
    print("--------------------------------")

    response = await openai_client.chat.completions.create(
        model=DEFAULT_REPORT_MODEL,
        messages=messages
    )
    return response.choices[0].message.content


async def process_link(link, user_query, search_query):
    print(f"正從以下網址獲取內容: {link}")
    page_text = await fetch_webpage_text(link)
    if not page_text:
        return None
    usefulness = await is_page_useful(user_query, page_text)
    print(f"  判斷網頁是否有用 {link} {'✅' if usefulness else '❌'}")
    if usefulness:
        context = await extract_relevant_context(user_query, search_query, page_text)
        if context:
            print(f"    從 {link} 提取的內容(前50字): {context[:50]}")
            return context
    return None

async def async_main():
    user_query = input("請輸入您的題目: ").strip()
    iter_limit_input = input("請輸入最大迭代次數(預設為 3): ").strip()
    iteration_limit = int(iter_limit_input) if iter_limit_input.isdigit() else 3

    aggregated_contexts = []
    all_search_queries = []
    iteration = 0

    # ----- INITIAL SEARCH QUERIES -----
    new_search_queries = await generate_first_search_queries(user_query)
    print("LLM 首次生成的搜尋查詢: ", new_search_queries)

    all_search_queries.extend(new_search_queries)

    # ----- ITERATIVE RESEARCH LOOP -----
    while iteration < iteration_limit:
        print(f"\n=== 第 {iteration + 1} 次迭代 ===")
        iteration_contexts = []

        # For each search query, perform searches concurrently.
        search_tasks = [perform_tavily_search(query) for query in new_search_queries]
        search_results = await asyncio.gather(*search_tasks)

        # Aggregate all unique links
        unique_links = {}
        for idx, links in enumerate(search_results):
            query = new_search_queries[idx]
            for link in links:
                if link not in unique_links:
                    unique_links[link] = query

        print(f"本次迭代共收集到 {len(unique_links)} 個唯一連結")

        # Process each link concurrently
        link_tasks = [
            process_link(link, user_query, unique_links[link])
            for link in unique_links
        ]
        link_results = await asyncio.gather(*link_tasks)

        # Collect non-None contexts
        for res in link_results:
            if res:
                iteration_contexts.append(res)

        if iteration_contexts:
            aggregated_contexts.extend(iteration_contexts)
        else:
            print("本次迭代中沒有找到有用的內容。")

        # ----- ASK THE LLM IF MORE SEARCHES ARE NEEDED -----
        new_search_queries = await get_new_search_queries(user_query, all_search_queries, aggregated_contexts)
        if len(new_search_queries) == 0:
            print("LLM 表示不需要進行更多研究")
            break
        else:
            print("LLM 提供了新的搜尋查詢：", new_search_queries)
            all_search_queries.extend(new_search_queries)

        iteration += 1

    # ----- FINAL REPORT -----
    print("\n正在生成最終報告...")
    final_report = await generate_final_report(user_query, aggregated_contexts)
    print("\n==== 最終報告 ====\n")
    print(final_report)


def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
