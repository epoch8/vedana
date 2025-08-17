import pytest
from jims_core.llms.llm_provider import LLMProvider, LLMSettings


@pytest.fixture(scope="module")
def vcr_config():
    return {"filter_headers": ["authorization"], "ignore_hosts": ["test"]}


# test that if we perform two requests through LLMProvider, the usage is aggregated correctly
@pytest.mark.asyncio
@pytest.mark.vcr
async def test_aggregate_usage():
    llm = LLMProvider(settings=LLMSettings(model="gpt-4.1-nano-2025-04-14"))

    await llm.chat_completion_plain(messages=[{"role": "user", "content": "Hello"}])
    await llm.chat_completion_plain(messages=[{"role": "user", "content": "World"}])

    usage = llm.usage
    assert usage.keys() == {"gpt-4.1-nano-2025-04-14"}

    assert usage["gpt-4.1-nano-2025-04-14"].requests_count == 2
    assert usage["gpt-4.1-nano-2025-04-14"].prompt_tokens > 0
    assert usage["gpt-4.1-nano-2025-04-14"].completion_tokens > 0
