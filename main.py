import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai.types.shared import reasoning
from pydantic import BaseModel, Field
import json
import jiter
from datetime import datetime

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(".env", override=True)

openai = AsyncOpenAI()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 掛載靜態文件目錄
app.mount("/static", StaticFiles(directory="static"), name="static")
    
## Form + Streaming
@app.get("/api/v1/completion_stream")
async def get_completion_stream(query: str):
    response = StreamingResponse(generate_completion_stream(query), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    return response

async def generate_completion_stream(query: str):
    response = await openai.chat.completions.create(
        model='gpt-4.1-mini',
        messages=[
            {"role": "system", "content": """You are a bilingual translation assistant. Follow these rules:
- If user inputs English, translate to Traditional Chinese (Taiwan variant)
- If user inputs Chinese, translate to English
- Provide only the translation without explanations unless requested
- Focus on accuracy and natural language expression"""},
            {'role': 'user', 'content': query}
        ],
        temperature=0,
        stream=True
    )

    async for chunk in response: 
      chunk_text = chunk.choices[0].delta.content or ""
      if chunk_text:
          yield f"data: {chunk_text}\n\n"
    
    yield "data: [DONE]\n\n"

## Form + Streaming + Structured Output

class QueryResult(BaseModel):
    content: str
    following_questions: list[str] = Field(description="3 follow-up questions exploring different aspects of the topic.")

@app.get("/api/v1/completion_json_stream")
async def get_completion_json_stream(query: str):
    response = StreamingResponse(generate_completion_json_stream(query), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    return response

async def generate_completion_json_stream(query: str):
    json_str = ''
    async with openai.beta.chat.completions.stream(model="gpt-4.1-mini", 
                                                   messages=[
                                                       {"role": "system", "content": """You are a helpful assistant that can answer questions and help with tasks. Always respond in Traditional Chinese."""},
                                                       {'role': 'user', 'content': query}
                                                   ],
                                                   stream_options={"include_usage": True}, 
                                                   response_format=QueryResult) as stream:
      previous_content = ""
      async for event in stream:
          if event.type == "content.delta":
              json_str += event.delta # 累加完整的 json 字串來做解析

              try:
                  result = jiter.from_json(json_str.encode('utf-8'), partial_mode="trailing-strings")
                  
                  # 兩種輸出方式:

                  # 方法一: 針對 content 欄位，我們只輸出 delta 新增的字: 前端做累加，例如 esponseContentDiv.innerHTML += jsonData.content;
                  if "content" in result:
                      current_content = result["content"]
                      if current_content != previous_content:
                          result["content"] = current_content[len(previous_content):]
                          previous_content = current_content
                      elif current_content == previous_content:
                          result["content"] = ''
                      
                  # 方法二: 其他欄位保持完整輸出: 前端要顯示的話，需要整個區塊替換掉

                  yield f"data: {json.dumps(result)}\n\n"
              except ValueError:
                  # JSON 還不完整，繼續等待更多數據
                  pass
          elif event.type == "chunk" and event.chunk.usage:
              usage = event.chunk.usage
              print(f"usage: {usage}")
              print(f"final output: {json_str}")
    
    done_event = { "message": "DONE" }
    yield f"data: {json.dumps(done_event)}\n\n"    
 

## Chat Agent + Streaming + Structured Output
from agents import Agent, Runner, set_default_openai_key, function_tool, trace, ItemHelpers, ModelSettings
from openai.types.responses import ResponseTextDeltaEvent, ResponseCompletedEvent

from tavily import AsyncTavilyClient

tavily_client = AsyncTavilyClient()

@function_tool
async def web_search(query: str) -> str:
    """Search the web for information"""
    print(f"  ⚙️ 呼叫函式 web_search 參數: {query}")

    result = await tavily_client.search(query)

    print(f"  ⚙️ 呼叫函式 web_search 結果: {result}")

    return result


@app.get("/api/v1/agent_stream")
async def get_agent_stream(query: str, previous_response_id: str = None, trace_id: str = None):
    response = StreamingResponse(generate_agent_stream(query, previous_response_id, trace_id), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    return response

async def generate_agent_stream(query: str, previous_response_id: str = None, trace_id: str = None):
    today = datetime.now().strftime("%Y-%m-%d")
    
    agent = Agent(
        name="QA Agent",
        instructions= f"""You are a helpful assistant that can answer questions and help with tasks. Always respond in Traditional Chinese. Today's date is {today}.""",
        tools=[web_search],
        model="gpt-5-mini",
        output_type=QueryResult,
        model_settings=ModelSettings(
            reasoning = {
                "effort": "low",
                "summary": "auto"
            }
        )
    )
    print(f"previous_response_id: {previous_response_id}")

    with trace("FastAPI Agent", trace_id=trace_id):
        result = Runner.run_streamed(agent, input=query, previous_response_id=previous_response_id)

    json_str = ''
    previous_content = ''
    last_response_id = None

    async for event in result.stream_events():
        print(event)
        if event.type == "raw_response_event" and event.data.type == "response.output_text.delta":
            print(event.data.delta)

            json_str += event.data.delta
            try:
                parsed_data = jiter.from_json(json_str.encode('utf-8'), partial_mode="trailing-strings")
                
                if "content" in parsed_data:
                    current_content = parsed_data["content"]
                    if current_content != previous_content:
                        parsed_data["content"] = current_content[len(previous_content):]
                        previous_content = current_content
                    elif current_content == previous_content:
                        parsed_data["content"] = ''
                    
                yield f"data: {json.dumps(parsed_data)}\n\n"
            except ValueError:
                # JSON 還不完整，繼續等待更多數據
                pass
        elif event.type == "raw_response_event" and event.data.type == "response.output_item.added" and event.data.item.type == "reasoning":
            think_chunk = {
                "message": "THINK_START",
            }
            yield f"data: {json.dumps(think_chunk)}\n\n"                                   
        elif event.type == "raw_response_event"  and event.data.type == "response.reasoning_summary_text.done":
            think_chunk = {
                "message": "THINK_TEXT",
                "text": event.data.text
            }
            yield f"data: {json.dumps(think_chunk)}\n\n"   
        elif event.type == "raw_response_event" and event.data.type == "response.completed":
            print(f"last_response_id: {event.data}")
            last_response_id = event.data.response.id
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
                yield f"data: {json.dumps({'message': 'CALL_TOOL', 'tool_name': str(event.item.raw_item.name), 'arguments': str(event.item.raw_item.arguments)})}\n\n"
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")                
            else:
                pass  # Ignore other event types            

    done_event = { "message": "DONE", "last_response_id": last_response_id }
    yield f"data: {json.dumps(done_event)}\n\n"    



## V2 版本: 增強 Context Engineering
from agents import SQLiteSession
# from agents.extensions.memory import AdvancedSQLiteSession
from custom_sqlite_session import CustomSQLiteSession

@app.get("/api/v2/agent_stream")
async def get_agent_stream_v2(query: str, thread_id: str):
    response = StreamingResponse(generate_agent_stream_v2(query, thread_id), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    return response

async def generate_agent_stream_v2(query: str, thread_id: str):

    # session = SQLiteSession(thread_id, "conversations.db")
    # session = AdvancedSQLiteSession(session_id=thread_id, create_tables=True, db_path="advanced_conversations.db")

    today = datetime.now().strftime("%Y-%m-%d")
    
    agent = Agent(
        name="QA Agent",
        instructions= f"""You are a helpful assistant that can answer questions and help with tasks. Always respond in Traditional Chinese. Today's date is {today}.""",
        tools=[web_search],
        model="gpt-5-mini",
        output_type=QueryResult,
        model_settings=ModelSettings(
            reasoning = {
                "effort": "low",
                "summary": "auto"
            }
        )
    )
    print(f"thread_id: {thread_id}")

    session = CustomSQLiteSession(thread_id, "conversations.db", agent=agent)
    current_items = await session.get_items()
    print(f"current_items_count: {len(current_items)}")

    with trace("FastAPI Agent", trace_id=f"trace_{thread_id}"):
        result = Runner.run_streamed(agent, input=query, session=session)

    json_str = ''
    previous_content = ''
    last_response_id = None
    
    async for event in result.stream_events():
        #print(event)
        if event.type == "raw_response_event" and event.data.type == "response.output_text.delta":
            #print(event.data.delta)

            json_str += event.data.delta
            try:
                parsed_data = jiter.from_json(json_str.encode('utf-8'), partial_mode="trailing-strings")
                
                if "content" in parsed_data:
                    current_content = parsed_data["content"]
                    if current_content != previous_content:
                        parsed_data["content"] = current_content[len(previous_content):]
                        previous_content = current_content
                    elif current_content == previous_content:
                        parsed_data["content"] = ''
                    
                yield f"data: {json.dumps(parsed_data)}\n\n"
            except ValueError:
                # JSON 還不完整，繼續等待更多數據
                pass
        elif event.type == "raw_response_event" and event.data.type == "response.output_item.added" and event.data.item.type == "reasoning":
            think_chunk = {
                "message": "THINK_START",
            }
            yield f"data: {json.dumps(think_chunk)}\n\n"                                   
        elif event.type == "raw_response_event"  and event.data.type == "response.reasoning_summary_text.done":
            think_chunk = {
                "message": "THINK_TEXT",
                "text": event.data.text
            }
            yield f"data: {json.dumps(think_chunk)}\n\n"   
        elif event.type == "raw_response_event" and event.data.type == "response.completed":
            print(f"last_response_id: {event.data}")
            last_response_id = event.data.response.id
        elif event.type == "run_item_stream_event":
            if event.item.type == "tool_call_item":
                print("-- Tool was called")
                yield f"data: {json.dumps({'message': 'CALL_TOOL', 'tool_name': str(event.item.raw_item.name), 'arguments': str(event.item.raw_item.arguments)})}\n\n"
            elif event.item.type == "tool_call_output_item":
                print(f"-- Tool output: {event.item.output}")
            elif event.item.type == "message_output_item":
                #print(f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}")                
                pass
            else:
                pass  # Ignore other event types            

    done_event = { "message": "DONE", "last_response_id": last_response_id }
    yield f"data: {json.dumps(done_event)}\n\n"

    print(f"result: {result.context_wrapper}")


