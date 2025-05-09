# https://github.com/openai/openai-agents-python/blob/main/examples/mcp/filesystem_example/main.py
# https://www.npmjs.com/package/@modelcontextprotocol/server-filesystem

from dotenv import load_dotenv
load_dotenv(".env", override=True)

import asyncio
import os
import shutil

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio

async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to read the filesystem and answer questions based on those files.",
        mcp_servers=[mcp_server],
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

    async with MCPServerStdio(
        name="Filesystem Server, via npx",
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", samples_dir],
        },
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Filesystem Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/{trace_id}\n")
            await run(server)


if __name__ == "__main__":
    # Let's make sure the user has npx installed
    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")

    asyncio.run(main())