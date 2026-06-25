import json
import re
from typing import cast

from fastmcp import Context
from vedana_core.app import VedanaApp
from vedana_core.rag_agent import row_to_text

from vedana_mcp.mcp import mcp

WRITE_PATTERN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP|LOAD)\b",
    re.IGNORECASE,
)


def strip_comments(query: str) -> str:
    query = re.sub(r"//[^\n]*", "", query)
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)
    return query


def is_write_query(query: str) -> bool:
    return bool(WRITE_PATTERN.search(strip_comments(query)))


@mcp.tool()
async def cypher_query(query: str, ctx: Context, limit: int = 50) -> str:
    """Execute a read-only Cypher query against the graph.

    Returns matched rows as JSON. Use limit to control how many rows come back (default 50).
    On error returns the error message so you can iterate on the query.
    """
    if is_write_query(query):
        return json.dumps(
            {"error": "Write queries are not allowed. Only read-only Cypher is permitted.", "query": query},
            ensure_ascii=False,
            indent=2,
        )

    app = cast(VedanaApp, ctx.request_context.lifespan_context)  # type: ignore[union-attr]
    try:
        records = list(await app.graph.execute_ro_cypher_query(query))
    except Exception as e:
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False, indent=2)

    total = len(records)
    truncated = total > limit
    rows = [row_to_text(r) for r in records[:limit]]

    return json.dumps(
        {
            "count": total,
            "truncated": truncated,
            "rows": rows,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool()
async def get_schema(ctx: Context) -> str:
    """Return the graph schema: node labels, edge types, and properties."""
    app = cast(VedanaApp, ctx.request_context.lifespan_context)  # type: ignore[union-attr]
    return await app.graph.llm_schema()


@mcp.tool()
async def get_stats(ctx: Context) -> str:
    """Return node and edge counts per label/type."""
    app = cast(VedanaApp, ctx.request_context.lifespan_context)  # type: ignore[union-attr]
    graph = app.graph

    nodes_by_label_records = await graph.execute_ro_cypher_query(
        "MATCH (n) UNWIND labels(n) AS lbl RETURN lbl, count(*) AS cnt ORDER BY cnt DESC"
    )
    edges_by_type_records = await graph.execute_ro_cypher_query(
        "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt ORDER BY cnt DESC"
    )

    nodes_by_label = {r["lbl"]: r["cnt"] for r in nodes_by_label_records}
    edges_by_type = {r["rel_type"]: r["cnt"] for r in edges_by_type_records}

    return json.dumps(
        {
            "nodes_total": sum(nodes_by_label.values()),
            "edges_total": sum(edges_by_type.values()),
            "nodes_by_label": nodes_by_label,
            "edges_by_type": edges_by_type,
        },
        ensure_ascii=False,
        indent=2,
    )
