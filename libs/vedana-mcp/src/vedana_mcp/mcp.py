from contextlib import asynccontextmanager

from fastmcp import FastMCP
from vedana_core.app import make_vedana_app

from vedana_mcp.auth import make_token_verifier


@asynccontextmanager
async def lifespan(server: FastMCP):
    yield await make_vedana_app()


mcp = FastMCP("vedana", lifespan=lifespan, auth=make_token_verifier())
