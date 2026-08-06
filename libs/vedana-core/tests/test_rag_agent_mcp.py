import json
import logging

import pytest

from vedana_core.rag_agent import RagAgent


class FakeExtraTool:
    name = "salary_ask"
    description = "Ask about salaries"
    openai_def = {"type": "function", "function": {"name": "salary_ask", "description": "Ask", "parameters": {}}}

    async def call(self, args_json: str) -> str:
        return "3500 EUR"


class FailingExtraTool:
    name = "salary_ask"
    description = "Ask about salaries"
    openai_def = {"type": "function", "function": {"name": "salary_ask", "description": "Ask", "parameters": {}}}

    async def call(self, args_json: str) -> str:
        raise RuntimeError("remote server is down")


class FakeLLM:
    """Captures the tools passed into generate_cypher_query_with_tools and calls the extra tool."""

    captured_tools = None

    async def generate_cypher_query_with_tools(self, data_descr, messages, tools, ctx):
        FakeLLM.captured_tools = tools
        # emulate the agentic tool-call loop invoking the extra tool
        for t in tools:
            if t.name == "salary_ask":
                await t.call(json.dumps({"question": "avg?"}))
        return [], "answer"

    async def generate_no_answer(self, dialog):
        return "no answer"


class FakeCtx:
    llm = None

    def context(self, n):
        return []


def _make_agent(extra, logger=None):
    return RagAgent(
        graph=None,
        vts=None,
        data_model_description="d",
        data_model_vts_indices=[],
        llm=FakeLLM(),
        ctx=FakeCtx(),
        extra_tools=extra,
        logger=logger,
    )


@pytest.mark.asyncio
async def test_extra_tools_passed_and_recorded():
    agent = _make_agent([FakeExtraTool()])
    answer, _events, _vts_queries, _cypher_queries, tool_calls = await agent.text_to_answer_with_vts_and_cypher(
        "q", threshold=0.8, top_n=10
    )
    assert answer == "answer"
    names = [t.name for t in FakeLLM.captured_tools]
    assert names == ["vector_text_search", "cypher", "salary_ask"]
    recorded = [c for c in tool_calls if c["tool"] == "salary_ask"]
    assert recorded == [{"tool": "salary_ask", "args": {"question": "avg?"}, "result": "3500 EUR"}]


@pytest.mark.asyncio
async def test_no_extra_tools_default():
    agent = _make_agent(None)
    await agent.text_to_answer_with_vts_and_cypher("q", threshold=0.8, top_n=10)
    assert [t.name for t in FakeLLM.captured_tools] == ["vector_text_search", "cypher"]


class SharedExtraTool:
    """Simulates a process-wide cached MCPTool instance (per load_mcp_tools):
    the same object is reused, unmutated, across multiple RagAgent instances."""

    name = "salary_ask"
    description = "Ask about salaries"
    openai_def = {"type": "function", "function": {"name": "salary_ask", "description": "Ask", "parameters": {}}}

    def __init__(self):
        self.calls = 0

    async def call(self, args_json: str) -> str:
        self.calls += 1
        return f"result-{self.calls}"


@pytest.mark.asyncio
async def test_shared_extra_tool_instance_not_mutated_across_invocations():
    """Regression: the recording wrapper must not monkeypatch the inner tool
    (extra_tools/MCPTool instances can be cached and reused across many
    text_to_answer_with_vts_and_cypher calls), otherwise .call gets re-wrapped every
    invocation and earlier invocations' tool_calls lists keep growing with later calls."""
    shared_tool = SharedExtraTool()
    original_call = shared_tool.call

    agent1 = _make_agent([shared_tool])
    _, _, _, _, tool_calls_1 = await agent1.text_to_answer_with_vts_and_cypher("q1", threshold=0.8, top_n=10)

    agent2 = _make_agent([shared_tool])
    _, _, _, _, tool_calls_2 = await agent2.text_to_answer_with_vts_and_cypher("q2", threshold=0.8, top_n=10)

    assert len(tool_calls_1) == 1
    assert len(tool_calls_2) == 1
    # invocation 1's list must not have grown after invocation 2 ran
    assert tool_calls_1 == [{"tool": "salary_ask", "args": {"question": "avg?"}, "result": "result-1"}]
    assert tool_calls_2 == [{"tool": "salary_ask", "args": {"question": "avg?"}, "result": "result-2"}]
    # the inner tool must never have been mutated (bound methods compare equal
    # when they wrap the same function on the same instance; `is` would always
    # be False for two separate attribute lookups even without any mutation)
    assert shared_tool.call == original_call
    assert shared_tool.call.__func__ is SharedExtraTool.call


@pytest.mark.asyncio
async def test_failing_extra_tool_call_is_logged(caplog):
    """Regression: unlike built-in tools (logged via
    LLM.create_completion_with_tools._execute_tool_call's logger.exception), MCP/extra tool
    failures went through _RecordingTool's bare `except Exception` and produced zero log
    output. A remote server erroring on every call must be visible in the logs, not silent."""
    agent = _make_agent([FailingExtraTool()], logger=logging.getLogger("test.rag_agent_mcp"))
    with caplog.at_level(logging.WARNING):
        answer, _events, _vts_queries, _cypher_queries, tool_calls = await agent.text_to_answer_with_vts_and_cypher(
            "q", threshold=0.8, top_n=10
        )

    # still recorded + returned as an error string (existing behavior preserved)
    recorded = [c for c in tool_calls if c["tool"] == "salary_ask"]
    assert len(recorded) == 1
    assert "remote server is down" in recorded[0]["result"]

    # ...but now also logged
    warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("salary_ask" in r.message and "remote server is down" in r.message for r in warnings)
