from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio
import asyncio

async def main():
    async with MCPServerStdio(
        params={
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-everything"],
        }
    ) as server:
        tools = await server.list_tools()
        for tool in tools:
            print(tool)
            print("--------------------------------")

if __name__ == "__main__":
    asyncio.run(main())