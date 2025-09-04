import asyncio

from jims_core.thread.thread_context import ThreadContext


async def say_hello(ctx: ThreadContext) -> None:
    await ctx.update_agent_status("thinking")
    await asyncio.sleep(1.5)  # Simulate some processing time
    ctx.send_message("Welcome to the simple pipeline!")


async def simple_pipeline(ctx: ThreadContext) -> None:
    await ctx.update_agent_status("thinking")
    res = await ctx.llm.chat_completion_plain(ctx.history)
    ctx.send_message(res.content if res.content else "No response from LLM")
