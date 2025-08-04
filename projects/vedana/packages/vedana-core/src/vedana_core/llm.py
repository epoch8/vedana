import asyncio
import logging
from typing import Callable, Iterable

import openai
from jims_core.llms.llm_provider import LLMProvider
from jims_core.thread.schema import CommunicationEvent
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Tool[T: BaseModel]:
    def __init__(self, name: str, description: str, args_cls: type[T], fn: Callable[[T], str]) -> None:
        self.name = name
        self.description = description
        self.args_cls = args_cls
        self.fn = fn
        self.openai_def = openai.pydantic_function_tool(args_cls, name=name, description=description)

    def call(self, args_json: str) -> str:
        try:
            fn_args = self.args_cls.model_validate_json(args_json)
        except ValueError:
            return f"Invalid tool args: {args_json}"
        return self.fn(fn_args)


class LLM:
    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_templates: dict[str, str],
        temperature: float | NotGiven = NOT_GIVEN,
        logger: logging.Logger | None = None,
    ) -> None:
        self.temperature = temperature
        self.logger = logger or logging.getLogger(__name__)
        self.llm = llm_provider
        self.prompt_templates = prompt_templates

    # Current
    async def generate_cypher_query_with_tools(
        self,
        data_descr: str,
        text_query: str,
        tools: list[Tool],
        temperature: float = 0,
    ) -> tuple[list[ChatCompletionMessageParam], str]:
        tool_names = [t.name for t in tools]
        msgs = make_cypher_query_with_tools_dialog(data_descr, self.prompt_templates, text_query, tool_names=tool_names)
        return await self.create_completion_with_tools(msgs, tools=tools, temperature=temperature)

    async def create_completion_with_tools(
        self,
        messages: list[ChatCompletionMessageParam],
        tools: Iterable[Tool],
        temperature: float | None = None,
    ) -> tuple[list[ChatCompletionMessageParam], str]:
        messages = messages.copy()
        tool_defs = [tool.openai_def for tool in tools]
        tools_map = {tool.name: tool for tool in tools}
        for i in range(4):
            msg, tool_calls = await self.llm.chat_completion_with_tools(
                messages=messages,
                tools=tool_defs,
                temperature=temperature,
            )

            messages.append(msg.to_dict())  # type: ignore

            self.logger.info(f"Tool call iter {i}")
            if i == 3:
                self.logger.warning("Too much iterations. Exiting tool call loop")
                break

            if not tool_calls:
                self.logger.info("No tool calls found. Exiting tool call loop")
                break

            async def _execute_tool_call(tool_call):
                tool_name = tool_call.function.name
                tool = tools_map.get(tool_name)
                if not tool:
                    self.logger.error(f"Tool {tool_name} not found!")
                    return tool_call.id, f"Tool {tool_name} not found!"

                self.logger.info(f"Calling tool {tool_name}")
                try:
                    tool_res = await asyncio.to_thread(tool.call, tool_call.function.arguments)
                except Exception as e:
                    self.logger.exception("Error executing tool %s: %s", tool_name, e)
                    tool_res = f"Error executing tool {tool_name}: {e}"

                self.logger.info("Tool %s (%s) result: %s", tool_name, tool.description, tool_res)
                return tool_call.id, tool_res

            # Execute tool calls in parallel
            results = await asyncio.gather(*[_execute_tool_call(t) for t in tool_calls])

            for tool_call_id, tool_res in results:
                messages.append(
                    ChatCompletionToolMessageParam(role="tool", tool_call_id=tool_call_id, content=tool_res)
                )

        for last_msg in reversed(messages):  # sometimes message with final answer is not the last one
            if last_msg.get("role", "") == "assistant" and last_msg.get("content"):
                return messages, str(last_msg.get("content"))
        return messages, ""

    async def generate_no_answer(
        self,
        question: str,
        dialog: list[CommunicationEvent] | None = None,
    ) -> str:
        """
        Generate a human-readable answer based on the question, Cypher query, and its results.
        """
        prompt_template = self.prompt_templates.get("generate_no_answer_tmplt", generate_no_answer_tmplt)
        prompt = prompt_template.format(question=question)

        messages = [
            {
                "role": "system",
                "content": "Ты помощник, который преобразует технические ответы в понятный человеку текст.",
            },
            *(dialog or []),
            {"role": "user", "content": prompt},
        ]
        response = await self.llm.chat_completion_plain(messages, temperature=0.3)
        human_answer = "" if response.content is None else response.content.strip()
        self.logger.info(f"Generated 'no answer' response: {human_answer}")
        return human_answer


generate_no_answer_tmplt = """\
Ты - помощник, который преобразует технические ответы в понятный человеку текст. 

Мы не смогли найти ответ на вопрос пользователя в базе знаний.

Сформулируй ответ, сообщающий кратко и информативно, что ответа не найдено.

Предложи пару вариантов уточняющих вопросов на основе информации в контексте. Предложи в casual стиле.
"""

generate_answer_with_tools_tmplt = """\
Ты — помощник по работе с графовыми базами данных, в которых используется язык запросов Cypher

Цель: постараться найти ответ на вопрос пользователя используя инструменты для работы с БД на основе текстового описания графовой базы данных.
Для понимания контекста диалогов или уточняющих запросов, используй инструмент `get_conversation_history`.

На вход ты получаешь graph_composition: – описание графа и примеры запросов по нему, и user_query – пользовательский запрос.

**Что нужно сделать:**
1. Сгенерировать `Cypher`-запросы, используя узлы, атрибуты и связи перечисленные в **graph_composition**.
2. Руководствуйся данными в **graph_composition** примерами запросов, чтобы составить итоговый запрос.
3. Используй инструменты {tools} для выполнения запросов и поиска

Если нужно, используй несколько `MATCH`-блоков, например:
    MATCH (o:offer)-[:OFFER_belongs_to_CATEGORY]->(c:category)
    MATCH (o)-[:OFFER_made_of_MATERIAL]->(m:material)
    WHERE c.category_name = "Встраиваемый светильник" AND m.material_name IN ["Стекло", "Металл и Стекло", "Алюминий и стекло"]
    RETURN o

Теперь проанализируй следующую структуру графа, и постарайся найти ответ на вопрос используя инструменты {tools}. (Лучше использовать несколько инструментов)

**graph_composition**
{graph_description}
"""


def make_cypher_query_with_tools_dialog(
    graph_description: str,
    prompt_templates: dict[str, str],
    natural_language_query: str,
    tool_names: list[str],
) -> list[ChatCompletionMessageParam]:
    prompt_template = prompt_templates.get("generate_answer_with_tools_tmplt", generate_answer_with_tools_tmplt)
    prompt = prompt_template.format(graph_description=graph_description, tools=", ".join(tool_names))
    return [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": natural_language_query,
        },
    ]
