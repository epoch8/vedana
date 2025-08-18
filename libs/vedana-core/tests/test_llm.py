import os

import litellm
import pytest
from jims_core.llms.llm_provider import LLMProvider
from pydantic import BaseModel

from vedana_core.llm import LLM, Tool

litellm_cache_dir = os.path.join(os.path.dirname(__file__), ".litellm_cache")
litellm.cache = litellm.Cache(type="disk", disk_cache_dir=litellm_cache_dir)
litellm._turn_on_debug()


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"], "ignore_hosts": ["test"]}


@pytest.mark.asyncio
async def test_llm_completion_with_tools() -> None:
    llm_provider = LLMProvider()
    llm = LLM(llm_provider=llm_provider, prompt_templates={})

    class HelloWorldArgs(BaseModel):
        name: str

    res_messages, res_content = await llm.create_completion_with_tools(
        messages=[
            {
                "role": "system",
                "content": "Call a tool 'hell_world' with argument 'name' set to 'Alice'.",
            },
        ],
        tools=[
            Tool(
                name="hello_world",
                description="A tool that says hello to the world.",
                args_cls=HelloWorldArgs,
                fn=lambda args: f"Hello, {args.name}!",
            ),
        ],
    )

    tool_call_msg = [msg for msg in res_messages if msg.get("role") == "tool"][0]
    assert tool_call_msg["content"] == "Hello, Alice!"
