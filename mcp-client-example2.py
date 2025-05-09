# https://github.com/openai/openai-agents-python/blob/main/examples/mcp/filesystem_example/main.py
# https://www.npmjs.com/package/@modelcontextprotocol/server-filesystem

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import asyncio
import os
import shutil
from contextlib import AsyncExitStack

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio

async def run(mcp_servers):
    agent = Agent(
        name="Assistant",
        instructions="""Based on the user's topic, decompose it into multiple search keywords, then use web_search to gather information and save the results in a search-{topic_name}.csv file.

## Detailed Process

1. Analyze the complexity of the user's topic and break it down into 2-5 distinct search keywords
2. For each keyword:
   - Execute a web search using the web_search tool
   - Record the search keyword, result URLs, and result summaries
3. Compile all search results into a CSV file with the following columns:
   - Search Keyword
   - Result URL
   - Result Summary
4. Save the file as search-{topic_name}.csv in the 'sample_files' directory
5. Report back to the user with:
   - The keywords you used
   - A brief summary of the search results
   - Confirmation that the CSV file has been created

## Response Guidelines

- Always respond to the user in Traditional Chinese (Taiwan)
- Use a professional but friendly tone
- Explain your keyword selection strategy
- Summarize key findings from the search
- Inform the user about any limitations encountered during the search process

## CSV Format Requirements

- UTF-8 encoding
- Proper escaping of special characters
- Headers in Traditional Chinese: "搜尋關鍵字", "網址", "摘要"
- Complete data for each field

Remember to focus on providing high-quality, relevant search results that best address the user's topic.
Always respond in Traditional Chinese (Taiwan).
""",
        mcp_servers=mcp_servers,
    )

    result = None
    while True:
        try:
            message = input("User: ")
            if message.lower() == 'exit':
                break

            if result:
                new_input = result.to_input_list() + [{"role": "user", "content": message}]
            else:
                new_input = [{"role": "user", "content": message}]

            result = await Runner.run(starting_agent=agent, input=new_input)
            print(f"AI: {result.final_output}")


        except KeyboardInterrupt:
            print("\n程式已中斷")
            break


async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    samples_dir = os.path.join(current_dir, "sample_files")

    servers = [
        {
            "name": "Filesystem Server, via npx",
            "params": {
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-filesystem", samples_dir],
            },
            "cls": MCPServerStdio,
        },
        {
            "name": "Tavily MCP Server, via npx",
            "params": {
                "command": "npx",
                "args": ["-y", "tavily-mcp@0.1.4"],
            },
            "cls": MCPServerStdio,
        }
    ]

    async with AsyncExitStack() as stack:
        mcp_servers = []
        for server in servers:
            ctx = server["cls"](name=server["name"], params=server["params"])
            mcp_server = await stack.enter_async_context(ctx)
            mcp_servers.append(mcp_server)

        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Tavily+Filesystem Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/{trace_id}\n")
            await run(mcp_servers)


if __name__ == "__main__":
    # Let's make sure the user has npx installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")

    asyncio.run(main())