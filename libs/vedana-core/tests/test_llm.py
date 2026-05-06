from unittest.mock import AsyncMock, MagicMock

import pytest
from jims_core.llms.llm_provider import LLMProvider
from pydantic import BaseModel

from vedana_core.llm import LLM, Tool


def mock_msg(content: str | None = None, tool_calls: list | None = None):
    """Create a mock message with to_dict() method."""
    m = MagicMock(content=content, tool_calls=tool_calls, role="assistant")
    m.to_dict.return_value = {"role": "assistant", "content": content}
    return m


def mock_tool_call(name: str, args: str, id: str = "call_1"):
    """Create a mock tool call."""
    tc = MagicMock(id=id)
    tc.function.name = name
    tc.function.arguments = args
    return tc


@pytest.mark.asyncio
async def test_llm_completion_with_tools() -> None:
    llm_provider = LLMProvider()

    tc = mock_tool_call("hello_world", '{"name": "Alice"}')
    llm_provider.chat_completion_with_tools = AsyncMock(
        side_effect=[(mock_msg(tool_calls=[tc]), [tc]), (mock_msg("Done!"), [])]
    )

    llm = LLM(llm_provider=llm_provider, prompt_templates={})

    class HelloWorldArgs(BaseModel):
        name: str

    res_messages, res_content = await llm.create_completion_with_tools(
        messages=[{"role": "system", "content": "Call hello_world with name='Alice'."}],
        tools=[
            Tool(
                name="hello_world", description="Says hello.", args_cls=HelloWorldArgs, fn=lambda a: f"Hello, {a.name}!"
            )
        ],
    )

    tool_msg = next(m for m in res_messages if m.get("role") == "tool")
    assert tool_msg.get("content") == "Hello, Alice!"


@pytest.mark.asyncio
async def test_llm_completion_no_tool_calls() -> None:
    llm_provider = LLMProvider()
    llm_provider.chat_completion_with_tools = AsyncMock(return_value=(mock_msg("No tools needed."), []))

    llm = LLM(llm_provider=llm_provider, prompt_templates={})

    class DummyArgs(BaseModel):
        value: str

    res_messages, res_content = await llm.create_completion_with_tools(
        messages=[{"role": "user", "content": "Hello"}],
        tools=[Tool(name="dummy", description="Dummy.", args_cls=DummyArgs, fn=lambda a: a.value)],
    )

    assert res_content == "No tools needed."
    assert not any(m.get("role") == "tool" for m in res_messages)
