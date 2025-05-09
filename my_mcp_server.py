# https://github.com/modelcontextprotocol/python-sdk

# mcp install server.py
# mcp dev server.py
from mcp.server.fastmcp import FastMCP

# Create an MCP server
mcp = FastMCP("Demo")

# Add an addition tool
@mcp.tool()
def generate_random_integer(a: int, b: int) -> int:
    """Generate a random integer between a and b (inclusive)"""
    import random
    return random.randint(a, b)

@mcp.resource("config://app")
def get_config() -> str:
    """Static configuration data"""
    return "This is App configuration"
    
@mcp.prompt()
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"    