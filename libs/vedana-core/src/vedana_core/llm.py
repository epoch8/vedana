import asyncio
import logging
from typing import Awaitable, Callable, Iterable

import openai
from jims_core.llms.llm_provider import LLMProvider
from jims_core.thread.schema import CommunicationEvent
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Tool[T: BaseModel]:
    def __init__(
        self, name: str, description: str, args_cls: type[T], fn: Callable[[T], Awaitable[str]] | Callable[[T], str]
    ) -> None:
        self.name = name
        self.description = description
        self.args_cls = args_cls
        self.fn = fn
        self.openai_def = openai.pydantic_function_tool(args_cls, name=name, description=description)

    async def call(self, args_json: str) -> str:
        try:
            fn_args = self.args_cls.model_validate_json(args_json)
        except ValueError:
            return f"Invalid tool args: {args_json}"

        if asyncio.iscoroutinefunction(self.fn):
            result = await self.fn(fn_args)
        else:
            result: str = await asyncio.to_thread(self.fn, fn_args)  # type: ignore

        return result


class LLM:
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_templates: dict[str, str],
        logger: logging.Logger | None = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.llm = llm_provider
        self.prompt_templates = prompt_templates

    # Current
    async def generate_cypher_query_with_tools(
        self,
        data_descr: str,
        messages: Iterable,
        tools: list[Tool],
    ) -> tuple[list[ChatCompletionMessageParam], str]:
        tool_names = [t.name for t in tools]
        msgs = make_cypher_query_with_tools_dialog(data_descr, self.prompt_templates, messages, tool_names=tool_names)
        return await self.create_completion_with_tools(msgs, tools=tools)

    async def create_completion_with_tools(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: Iterable[Tool],
    ) -> tuple[list[ChatCompletionMessageParam], str]:
        messages = messages.copy()
        tool_defs = [tool.openai_def for tool in tools]
        tools_map = {tool.name: tool for tool in tools}

        async def _execute_tool_call(tool_call):
            tool_name = tool_call.function.name
            tool = tools_map.get(tool_name)
            if not tool:
                self.logger.error(f"Tool {tool_name} not found!")
                return tool_call.id, f"Tool {tool_name} not found!"

            self.logger.debug(f"Calling tool {tool_name}")
            try:
                tool_res = await tool.call(tool_call.function.arguments)
            except Exception as e:
                self.logger.exception("Error executing tool %s: %s", tool_name, e)
                tool_res = f"Error executing tool {tool_name}: {e}"

            self.logger.debug("Tool %s (%s) result: %s", tool_name, tool.description, tool_res)
            return tool_call.id, tool_res

        max_iters = 5
        for i in range(max_iters):
            msg, tool_calls = await self.llm.chat_completion_with_tools(
                messages=messages,
                tools=tool_defs,
            )

            messages.append(msg.to_dict())  # type: ignore

            if not tool_calls:
                self.logger.debug("No tool calls found. Exiting tool call loop")
                break

            self.logger.debug(f"Tool call iter {i + 1}/{max_iters}")

            # Execute tool calls in parallel
            results = await asyncio.gather(*[_execute_tool_call(t) for t in tool_calls])

            for tool_call_id, tool_res in results:
                messages.append(
                    ChatCompletionToolMessageParam(role="tool", tool_call_id=tool_call_id, content=tool_res)
                )

            if i == max_iters - 1:
                self.logger.warning(f"Reached tool call iteration limit ({max_iters}). Exiting tool call loop")
                finalize_prompt = self.prompt_templates.get("finalize_answer_tmplt", finalize_answer_tmplt)
                finalize_msg = {"role": "system", "content": finalize_prompt}
                final_msg = await self.llm.chat_completion_plain(messages + [finalize_msg])
                messages.append(final_msg.to_dict())  # type: ignore
                break

        for last_msg in reversed(messages):  # sometimes message with final answer is not the last one
            if last_msg.get("role", "") == "assistant" and last_msg.get("content"):
                return messages, str(last_msg.get("content"))
        return messages, ""

    async def generate_no_answer(
        self,
        dialog: list[CommunicationEvent] | None = None,
    ) -> str:
        prompt = self.prompt_templates.get("generate_no_answer_tmplt", generate_no_answer_tmplt)
        messages = [
            {"role": "system", "content": prompt},
            *(dialog or []),
        ]
        response = await self.llm.chat_completion_plain(messages)
        human_answer = "" if response.content is None else response.content.strip()
        self.logger.debug(f"Generated 'no answer' response: {human_answer}")
        return human_answer


finalize_answer_tmplt = """\
Formulate an answer to the user's request based on the information obtained from tool call results.
If the information is not sufficient for an accurate answer, clearly describe the limitations and suggest 1-2 clarifying questions.
Important: do not explicitly mention tools; refer only to the data.
"""

generate_no_answer_tmplt = """\
You are an assistant that turns technical outputs into human-friendly text.
We could not find an answer to the user's question in the knowledge base.
Write a brief and informative response saying that no answer was found.
Suggest a couple of clarifying follow-up questions based on the available context. Use a casual tone.
"""

generate_answer_with_tools_tmplt = """\
You are an assistant for graph databases that use the Cypher query language.

Goal: try to find an answer to the user's question by using database tools based on a text description of the graph database.

Input:
- `graph_composition`: graph description and example queries
- `user_query`: the user's request

**What you need to do:**
1. Generate `Cypher` queries using the nodes, attributes, and relationships listed in **graph_composition**.
2. Use the data and query examples in **graph_composition** to build the final query.
3. Use tools {tools} to execute queries and search for the answer.

If needed, use multiple `MATCH` blocks, for example:
    MATCH (o:offer)-[:OFFER_belongs_to_CATEGORY]->(c:category)
    MATCH (o)-[:OFFER_made_of_MATERIAL]->(m:material)
    WHERE c.category_name = "fixture" AND m.material_name IN ["Glass", "Metal"]
    RETURN o

Now analyze the following graph structure and try to find the answer using tools {tools}. (Using multiple tools is preferred.)

**graph_composition:**
{graph_description}
"""


def make_cypher_query_with_tools_dialog(
    graph_description: str,
    prompt_templates: dict[str, str],
    messages: Iterable,
    tool_names: list[str],
) -> list[ChatCompletionMessageParam]:
    prompt_template = prompt_templates.get("generate_answer_with_tools_tmplt", generate_answer_with_tools_tmplt)
    prompt = prompt_template.format(graph_description=graph_description, tools=", ".join(tool_names))
    return [
        {
            "role": "system",
            "content": prompt,
        },
        *messages,
    ]
