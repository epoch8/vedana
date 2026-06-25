import datetime
import json
from typing import cast

from fastmcp import Context
from jims_core.llms.llm_provider import LLMProvider
from jims_core.thread.schema import CommunicationEvent, EventEnvelope
from jims_core.thread.thread_context import ThreadContext
from jims_core.util import uuid7
from vedana_core.app import VedanaApp
from vedana_core.llm import LLM
from vedana_core.rag_agent import RagAgent
from vedana_core.settings import settings

from vedana_mcp.mcp import mcp


def make_thread_ctx(question: str) -> ThreadContext:
    thread_id = uuid7()
    user_event: EventEnvelope = EventEnvelope(
        thread_id=thread_id,
        event_id=uuid7(),
        created_at=datetime.datetime.now(),
        event_type="comm.user_message",
        event_data=CommunicationEvent(role="user", content=question),
    )
    return ThreadContext(
        thread_id=thread_id,
        history=[CommunicationEvent(role="user", content=question)],
        events=[user_event],
        llm=LLMProvider(),
    )


@mcp.resource("vedana://data-model/description")
async def data_model_description(ctx: Context) -> str:
    """Full data model description: nodes, attributes, links, queries."""
    app = cast(VedanaApp, ctx.request_context.lifespan_context)  # type: ignore[union-attr]
    return await app.data_model.to_text_descr()


@mcp.resource("vedana://data-model/schema")
async def data_model_schema(ctx: Context) -> str:
    """Compact data model schema as JSON: nodes, links, query scenarios."""
    app = cast(VedanaApp, ctx.request_context.lifespan_context)  # type: ignore[union-attr]
    schema = await app.data_model.to_compact_json()
    return json.dumps(schema, ensure_ascii=False, indent=2)


@mcp.tool()
async def ask_vedana(question: str, ctx: Context) -> str:
    """Ask a question about legislation. Returns the answer and the retrieval tool calls made."""
    app = cast(VedanaApp, ctx.request_context.lifespan_context)  # type: ignore[union-attr]

    dm = app.data_model
    data_model_desc = await dm.to_text_descr()
    vts_indices = await dm.vector_indices()
    prompt_templates = await dm.prompt_templates()

    thread_ctx = make_thread_ctx(question)
    llm = LLM(thread_ctx.llm, prompt_templates=prompt_templates)

    agent = RagAgent(
        graph=app.graph,
        vts=app.vts,
        data_model_description=data_model_desc,
        data_model_vts_indices=vts_indices,
        llm=llm,
        ctx=thread_ctx,
    )

    answer, _, _, _, tool_calls = await agent.text_to_answer_with_vts_and_cypher(
        question,
        threshold=settings.embeddings_threshold,
        top_n=settings.embeddings_top_n,
    )

    return json.dumps(
        {"answer": answer, "tool_calls": tool_calls},
        ensure_ascii=False,
        indent=2,
    )
