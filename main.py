import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import json
import jiter

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
              json_str += event.delta
              try:
                  result = jiter.from_json(json_str.encode('utf-8'), partial_mode="trailing-strings")
                  
                  # 針對 content 只輸出 increment 最後文字
                  if "content" in result:
                      current_content = result["content"]
                      if current_content != previous_content:
                          result["content"] = current_content[len(previous_content):]
                          previous_content = current_content
                      elif current_content == previous_content:
                          result["content"] = ''
                      
                  yield f"data: {json.dumps(result)}\n\n"
              except ValueError:
                  # JSON 還不完整，繼續等待更多數據
                  pass
          elif event.type == "chunk" and event.chunk.usage:
              usage = event.chunk.usage      
              streaming_results["input"] = messages     
              streaming_results["output"] = json.loads(json_str)                
              streaming_results["metrics"]["prompt_tokens"] = usage.prompt_tokens
              streaming_results["metrics"]["completion_tokens"] = usage.completion_tokens
              streaming_results["metrics"]["total_tokens"] = usage.total_tokens
              yield(streaming_results)
    
    yield "data: [DONE]\n\n"    



## Chat Agent + Streaming + Structured Output
