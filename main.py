import os
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(".env", override=True)

openai = OpenAI()
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
    response = openai.chat.completions.create(
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

    for chunk in response: 
      chunk_text = chunk.choices[0].delta.content or ""
      if chunk_text:
          yield f"data: {chunk_text}\n\n"
    
    yield "data: [DONE]\n\n"

## Form + Streaming + Structured Output

@app.get("/api/v1/completion_json_stream")
async def get_completion_json_stream(query: str):
    response = StreamingResponse(generate_completion_json_stream(query), media_type="text/event-stream")
    response.headers["X-Accel-Buffering"] = "no"
    return response

async def generate_completion_json_stream(query: str):
    response = openai.chat.completions.create(
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

    for chunk in response: 
      chunk_text = chunk.choices[0].delta.content or ""
      if chunk_text:
          yield f"data: {chunk_text}\n\n"
    
    yield "data: [DONE]\n\n"    



## Chat Agent + Streaming + Structured Output
