import os

from vedana_mcp.mcp import mcp
import vedana_mcp.tools_rag  # noqa: F401
import vedana_mcp.tools_graph  # noqa: F401


def main() -> None:
    mcp.run(
        transport="streamable-http",
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "8000")),
    )
