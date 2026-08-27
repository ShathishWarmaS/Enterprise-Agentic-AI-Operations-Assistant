"""Expose the project's tools over the Model Context Protocol (stdio).

    python scripts/mcp_server.py

This lets an MCP client (Claude Desktop, another agent) call the same
`search_documents` / `query_data` / `generate_checklist` tools the Action agent
uses. It reuses `ToolRegistry`, so there is exactly one implementation of each
tool. Requires the optional `mcp` package:

    uv pip install mcp
"""

from __future__ import annotations

import asyncio
import json

from app.data.database import init_db
from app.services.container import get_container


async def _serve() -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent
        from mcp.types import Tool as MCPTool
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("the 'mcp' package is required: uv pip install mcp") from exc

    init_db()
    registry = get_container().tools
    server: Server = Server("enterprise-agentic-ai")

    @server.list_tools()
    async def list_tools() -> list[MCPTool]:
        return [
            MCPTool(name=t.name, description=t.description, inputSchema=t.input_schema)
            for t in registry.all()
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        call = registry.get(name).invoke(arguments)
        payload = call.result if call.ok else {"error": call.error}
        return [TextContent(type="text", text=json.dumps(payload, indent=2, default=str))]

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(_serve())
