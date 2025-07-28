import json
import logging
import re
from collections import defaultdict
from typing import Any, Callable, Iterable, Type, TypeVar
import asyncio

import openai
from jims_core.llms.llm_provider import LLMProvider
from openai import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionToolMessageParam,
)
from pydantic import BaseModel

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=BaseModel)


class Tool:
    def __init__(self, name: str, description: str, args_cls: Type[T], fn: Callable[[T], str]) -> None:
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

    async def generate_cypher_query(self, data_descr: str, text_query: str) -> str:
        return await generate_cypher_query_v4(self.llm, self.prompt_templates, data_descr, text_query)

    async def generate_cypher_query_v5(self, data_descr: str, text_query: str) -> str:
        return await generate_cypher_query_v5(self.llm, self.prompt_templates, data_descr, text_query)

    async def generate_cypher_query_v5_with_tools(
        self,
        data_descr: str,
        text_query: str,
        tools: list[Tool],
        temperature: float = 0,
    ) -> tuple[list[ChatCompletionMessageParam], str]:
        msgs = make_cypher_query_v5_with_tools_dialog(data_descr, self.prompt_templates, text_query)
        return await self.create_completion_with_tools(msgs, tools=tools, temperature=temperature)

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
        temperature: float | NotGiven = NOT_GIVEN,
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

    async def extract_attributes_from_cypher(self, cypher_query: str) -> dict[str, Any]:
        """
        Извлекает атрибуты и их значения из Cypher-запроса с помощью LLM.
        Возвращает словарь, где значения могут быть списками, если ключ повторяется.
        """
        self.logger.debug("🔍 Parsing Cypher query:")
        self.logger.debug(cypher_query)

        prompt = self.prompt_templates.get(
            "extract_attributes_from_cypher_tmplt", extract_attributes_from_cypher_tmplt
        ).format(cypher_query=cypher_query)

        messages: list[ChatCompletionMessageParam] = [
            # TODO отрефакторить промпты в корректный формат - системный промпт-инструкция и юзерский - контент.
            {"role": "system", "content": "Ты помощник по работе с базами данных."},
            {"role": "user", "content": prompt},
        ]
        response = await self.llm.chat_completion_plain(messages)  # todo parse format а не костыли ниже
        raw_response = content_from_completion(response)

        self.logger.debug("🧐 LLM returned (raw):")
        self.logger.debug(raw_response)

        # Remove wrapping ```json and ``` blocks
        cleaned = re.sub(r"```json\s*", "", raw_response, flags=re.IGNORECASE)
        cleaned = re.sub(r"```", "", cleaned).strip()

        self.logger.debug("🧼 Cleaned JSON block:")
        self.logger.debug(cleaned)

        try:
            parsed = json.loads(cleaned)

            if isinstance(parsed, dict):
                self.logger.debug("✅ Extracted dictionary:")
                self.logger.debug(parsed)
                return parsed

            elif isinstance(parsed, list) and all(isinstance(item, dict) for item in parsed):
                merged = defaultdict(list)
                for d in parsed:
                    for key, value in d.items():
                        merged[key].append(value)
                self.logger.debug("✅ Extracted and merged list of dictionaries:")
                self.logger.debug(merged)
                return dict(merged)

            else:
                self.logger.warning("⚠️ JSON is valid, but the structure does not match expectations.")
                return {}

        except json.JSONDecodeError as e:
            self.logger.warning(f"⚠️ JSON parsing error: {e}")
            return {}

    async def filter_graph_structure(self, graph_descr: str, natural_language_query: str) -> str:
        """
        Inspect the graph structure (in text form) and the natural language query, leaving only the required nodes/attributes/links.
        """
        self.logger.debug(f"🔹 Filtering graph structure for query {natural_language_query}")
        self.logger.debug(f"🔹 Full graph structure:\n{graph_descr}\n")

        prompt_template = self.prompt_templates.get("filter_graph_structure_tmplt", filter_graph_structure_tmplt)
        prompt = prompt_template.format(graph_composition=graph_descr, natural_language_query=natural_language_query)

        messages: list[ChatCompletionMessageParam] = [
            # TODO отрефакторить промпты в корректный формат - системный промпт-инструкция и юзерский - контент.
            {"role": "system", "content": "Ты — помощник по работе с графовыми базами данных."},
            {"role": "user", "content": prompt},
        ]
        response = await self.llm.chat_completion_plain(messages)
        response_text = content_from_completion(response)

        self.logger.debug(f"🔹 Filtered graph structure:\n{response_text}\n")
        return response_text

    async def generate_human_answer(
        self,
        question: str,
        query_result: str,
        dialog: list[ChatCompletionMessageParam] | None = None,
    ) -> str:
        """
        Generate a human-readable answer based on the question, Cypher query, and its results.
        """
        prompt_template = self.prompt_templates.get("generate_human_answer_tmplt", generate_human_answer_tmplt)
        prompt = prompt_template.format(question=question, query_result=query_result)

        messages: list[ChatCompletionMessageParam] = [
            # TODO отрефакторить промпты в корректный формат - системный промпт-инструкция и юзерский - контент.
            {
                "role": "system",
                "content": "Ты помощник, который преобразует технические ответы в понятный человеку текст.",
            },
            *(dialog or []),
            {"role": "user", "content": prompt},
        ]
        response = await self.llm.chat_completion_plain(messages, temperature=0.3)
        human_answer = content_from_completion(response)
        self.logger.info(f"Generated human answer: {human_answer}")
        return human_answer

    async def generate_no_answer(
        self,
        question: str,
        dialog: list[ChatCompletionMessageParam] | None = None,
    ) -> str:
        """
        Generate a human-readable answer based on the question, Cypher query, and its results.
        """
        prompt_template = self.prompt_templates.get("generate_no_answer_tmplt", generate_no_answer_tmplt)
        prompt = prompt_template.format(question=question)

        messages: list[ChatCompletionMessageParam] = [
            # TODO отрефакторить промпты в корректный формат - системный промпт-инструкция и юзерский - контент.
            {
                "role": "system",
                "content": "Ты помощник, который преобразует технические ответы в понятный человеку текст.",
            },
            *(dialog or []),
            {"role": "user", "content": prompt},
        ]
        response = await self.llm.chat_completion_plain(messages, temperature=0.3)
        human_answer = content_from_completion(response)
        self.logger.info(f"Generated 'no answer' response: {human_answer}")
        return human_answer

    async def update_cypher_with_alt_values(
        self, text_query: str, cypher_query: str, alternative_values: dict[str, set]
    ) -> str:
        prompt_template = self.prompt_templates.get(
            "update_cypher_with_alt_values_tmplt", update_cypher_with_alt_values_tmplt
        )
        refine_prompt = prompt_template.format(
            natural_language_query=text_query,
            cypher_query=cypher_query,
            alternative_values=alternative_values,
        )

        messages: list[ChatCompletionMessageParam] = [
            # TODO отрефакторить промпты в корректный формат - системный промпт-инструкция и юзерский - контент.
            {
                "role": "system",
                "content": "Ты — помощник по работе с графовыми базами данных, в которых используется подмножество Cypher, совместимое с NetworkX.",
            },
            {"role": "user", "content": refine_prompt},
        ]
        response = await self.llm.chat_completion_plain(messages)

        cypher_query = cypher_from_completion(response)
        self.logger.debug(f"🔹 Updated Cypher query:\n{cypher_query}\n")
        return cypher_query


extract_attributes_from_cypher_tmplt = """\
У нас есть Cypher-запрос:
{cypher_query}

Выдели все пары {{атрибут: значение}}, которые используются в WHERE.
Верни ТОЛЬКО JSON, без пояснений.

Пример:
Cypher-запрос:
MATCH (n:category {{category_name: "встраиваемые светильники"}})
WHERE n.category_type = "technical"
RETURN n

Ожидаемый ответ:
{{
    "category_name": "встраиваемые светильники",
    "category_type": "technical"
}}
"""


filter_graph_structure_tmplt = """\
У нас есть описанный граф знаний и запрос пользователя на естественном языке.
Граф знаний состоит из узлов и связей, у каждого узла может быть несколько атрибутов.

Твоя задача:
Проанализируй пользовательский запрос, структуру графа, и убери из структуры все строки, которые нерелевантны запросу.
Обрати внимание, что у некоторых узлов могут быть атрибуты, которые обязательно нужно оставить если в запросе будет использоваться данный узел.

Структура графа:
{graph_composition}

Пользовательский запрос:
{natural_language_query}

В ответе верни только верни только отфильтрованные строки, больше ничего не возвращай и не изменяй.
"""

generate_human_answer_tmplt = """\
Вопрос пользователя: {question}

Результаты запросов:
{query_result}

Сформулируй понятный человеку ответ на русском языке на основе этих данных.
Ответ должен быть кратким, но информативным.
Используй bullet points, максимально упрощая восприятие.
Не упоминай Cypher-запрос или технические детали в ответе.

Что могут спрашивать:
- какие светильники соответствуют заданным требованиям
-- перечисли наиболее подходящие товары из ответа

- какие характеристики у товара
-- приведи описание товара

Предложи пару вариантов уточняющих вопросов на основе информации в контексте, которая не вошла в ответ. Предложи в casual стиле, не пиши что это уточняющий вопрос.
"""

generate_no_answer_tmplt = """\
Вопрос пользователя: {question}

Мы не смогли найти ответ на данный вопрос в базе знаний.

Сформулируй ответ, сообщающий кратко и информативно, что ответа не найдено.

Предложи пару вариантов уточняющих вопросов на основе информации в контексте. Предложи в casual стиле.
"""

update_cypher_with_alt_values_tmplt = """\
Ты помогаешь уточнять Cypher-запросы на основе пользовательских запросов и списка альтернативных значений.

От пользователя был получен следующий запрос:
"{natural_language_query}"

На основе него мы создали такой Cypher-запрос:
{cypher_query}

Мы нашли альтернативные значения для некоторых атрибутов:
{alternative_values}

Действуй строго по следующим правилам:
1. Если в запросе пользователя явно указано одно конкретное значение, выбери только одно наиболее подходящее значение из списка альтернативных значений и подставь его в условие WHERE ... = "...".
2. Если из текста запроса неясно, какое значение имелось в виду, и у атрибута есть список альтернатив — перепиши запрос, заменив точное сравнение на IN [...], включая все возможные значения.

Если нужно, используй несколько `MATCH`-блоков, например:
    MATCH (o:offer)-[:OFFER_belongs_to_CATEGORY]->(c:category)
    MATCH (o)-[:OFFER_made_of_MATERIAL]->(m:material)
    WHERE c.category_name = "Встраиваемый светильник" AND m.material_name IN ["Стекло", "Металл и Стекло", "Алюминий и стекло"]
    RETURN o

Верни ТОЛЬКО обновленный Cypher-запрос или массив cypher-запросов, в том же виде, в котором они поступили на вход больше ничего не говори.
Везде используй "
"""


def content_from_completion(completion: ChatCompletionMessage) -> str:
    if completion.content is None:
        return ""
    return completion.content.strip() or ""


def clear_cypher(cypher: str) -> str:
    return cypher.strip().removeprefix("""```cypher""").removeprefix("""```""").removesuffix("```").strip()


def cypher_from_completion(completion: ChatCompletionMessage) -> str:
    return clear_cypher(content_from_completion(completion))


generate_cypher_query_template_v4 = """\
Ты — помощник по работе с графовыми базами данных, в которых используется подмножество Cypher, совместимое с NetworkX.

Цель: сгенерировать **НЕСКОЛЬКО корректных Cypher-запросов** на основе текстового описания графовой базы данных и запроса пользователя.

На вход ты получаешь graph_composition: – описание графа и примеры запросов по нему, и user_query – пользовательский запрос.

**Что нужно сделать:**
1. Сгенерировать `Cypher`-запросы, используя узлы, атрибуты и связи перечисленные в **graph_composition**.
2. Руководствуйся данными в **graph_composition** примерами запросов, чтобы составить итоговый запрос.
3. Используй только допустимые конструкции из списка выше
5. Не добавляй пояснений или обёрток — верни только валидные Cypher-запросы
6. Используй везде двойные кавычки "
7. Каждый запрос должен полностью отвечать на вопрос пользователя. Допустимо вернуть массив из нескольких запросов, если пользователь спрашивает про несколько разных узлов одного типа (например, запрос типа "сравни/в чем разница")
8. MATCH-блоки ОБЯЗАТЕЛЬНО пиши один за другим, не разделяй WHERE.
9. В одном запросе должен быть СТРОГО ОДИН блок WHERE

Если нужно, используй несколько `MATCH`-блоков, например:
    MATCH (o:offer)-[:OFFER_belongs_to_CATEGORY]->(c:category)
    MATCH (o)-[:OFFER_made_of_MATERIAL]->(m:material)
    WHERE c.category_name = "Встраиваемый светильник" AND m.material_name IN ["Стекло", "Металл и Стекло", "Алюминий и стекло"]
    RETURN o

Теперь проанализируй следующую структуру графа, и преобразуй пользовательский запрос в Cypher запросы.

ВЕРНИ ОТВЕТ В ВИДЕ строк, разделённых знаком "---", пример:
MATCH (n:Product) RETURN n LIMIT 1
---
MATCH (m:Vendor) RETURN m LIMIT 1

**graph_composition**
{filtered_graph}

**user_query**
{natural_language_query}

ВЕРНИ ОТВЕТ В ВИДЕ строк, разделённых знаком "---", пример:
MATCH (n:Product) RETURN n LIMIT 1
---
MATCH (m:Vendor) RETURN m LIMIT 1
"""


async def generate_cypher_query_v4(
    llm: LLMProvider,
    prompt_templates: str,
    filtered_graph: str,
    natural_language_query: str
):
    prompt_template = prompt_templates.get("generate_cypher_query_template_v4", generate_cypher_query_template_v4)
    prompt = prompt_template.format(filtered_graph=filtered_graph, natural_language_query=natural_language_query)
    messages: list[ChatCompletionMessageParam] = [
        # TODO отрефакторить промпты в корректный формат - системный промпт-инструкция и юзерский - контент.
        {"role": "system", "content": "Ты — помощник по работе с графовыми базами данных."},
        {"role": "user", "content": prompt},
    ]
    response_cypher = await llm.chat_completion_plain(messages)
    cypher_query = cypher_from_completion(response_cypher)
    return cypher_query


generate_cypher_query_template_v5 = """\
Ты — помощник по работе с графовыми базами данных, в которых используется язык запросов Cypher

Цель: сгенерировать **НЕСКОЛЬКО корректных Cypher-запросов и запросов для текстового поиска** на основе текстового описания графовой базы данных и запроса пользователя.

На вход ты получаешь graph_composition: – описание графа и примеры запросов по нему, и user_query – пользовательский запрос.

**Что нужно сделать:**
1. Сгенерировать `Cypher`-запросы, используя узлы, атрибуты и связи перечисленные в **graph_composition**.
2. Руководствуйся данными в **graph_composition** примерами запросов, чтобы составить итоговый запрос.
3. Не добавляй пояснений или обёрток — верни только валидные Cypher-запросы и запросы для текстового поиска
4. Каждый запрос должен полностью отвечать на вопрос пользователя. Допустимо вернуть массив из нескольких запросов, если пользователь спрашивает про несколько разных узлов одного типа (например, запрос типа "сравни/в чем разница")

Если нужно, используй несколько `MATCH`-блоков, например:
    MATCH (o:offer)-[:OFFER_belongs_to_CATEGORY]->(c:category)
    MATCH (o)-[:OFFER_made_of_MATERIAL]->(m:material)
    WHERE c.category_name = "Встраиваемый светильник" AND m.material_name IN ["Стекло", "Металл и Стекло", "Алюминий и стекло"]
    RETURN o

Дополнительно можно выполнять текстовый поиск используя синтаксис:
    vector_search("${{node_label}}", "${{attribute_name}}", "${{text_query}}")
Например:
    vector_search("document", "text", "пшеница зерновая");

Теперь проанализируй следующую структуру графа, и преобразуй пользовательский запрос в Cypher запросы и запросы для текстового поиска.

**graph_composition**
{filtered_graph}

ВЕРНИ ОТВЕТ В ВИДЕ строк, разделённых знаком "---", пример:
MATCH (n:Product) RETURN n LIMIT 1
---
MATCH (m:Vendor) RETURN m LIMIT 1
---
vector_search("document", "text", "пшеница зерновая");

"""


async def generate_cypher_query_v5(
    llm: LLMProvider,
    prompt_templates: dict[str, str],
    filtered_graph: str,
    natural_language_query: str,
):
    prompt_template = prompt_templates.get("generate_cypher_query_template_v5", generate_cypher_query_template_v5)
    prompt = prompt_template.format(filtered_graph=filtered_graph)
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": natural_language_query,
        },
    ]
    response_cypher = await llm.chat_completion_plain(messages)
    cypher_query = cypher_from_completion(response_cypher)
    return cypher_query


generate_cypher_query_template_v5_with_tools = """\
Ты — помощник по работе с графовыми базами данных, в которых используется язык запросов Cypher

Цель: сгенерировать **НЕСКОЛЬКО корректных Cypher-запросов** на основе текстового описания графовой базы данных и запроса пользователя.

На вход ты получаешь graph_composition: – описание графа и примеры запросов по нему, и user_query – пользовательский запрос.

**Что нужно сделать:**
1. Сгенерировать `Cypher`-запросы, используя узлы, атрибуты и связи перечисленные в **graph_composition**.
2. Руководствуйся данными в **graph_composition** примерами запросов, чтобы составить итоговый запрос.
3. Не добавляй пояснений или обёрток — верни только валидные Cypher-запросы
4. Каждый запрос должен полностью отвечать на вопрос пользователя. Допустимо вернуть массив из нескольких запросов, если пользователь спрашивает про несколько разных узлов одного типа (например, запрос типа "сравни/в чем разница")
5. Используй при необходимости инструмент vector_text_search

Если нужно, используй несколько `MATCH`-блоков, например:
    MATCH (o:offer)-[:OFFER_belongs_to_CATEGORY]->(c:category)
    MATCH (o)-[:OFFER_made_of_MATERIAL]->(m:material)
    WHERE c.category_name = "Встраиваемый светильник" AND m.material_name IN ["Стекло", "Металл и Стекло", "Алюминий и стекло"]
    RETURN o

Теперь проанализируй следующую структуру графа, и преобразуй пользовательский запрос в Cypher запросы и запросы для текстового поиска.

**graph_composition**
{filtered_graph}

ВЕРНИ ОТВЕТ В ВИДЕ строк, разделённых знаком "---", пример:
MATCH (n:Product) RETURN n LIMIT 1
---
MATCH (m:Vendor) RETURN m LIMIT 1
"""


def make_cypher_query_v5_with_tools_dialog(
    filtered_graph: str,
    prompt_templates: dict[str, str],
    natural_language_query: str,
) -> list[ChatCompletionMessageParam]:
    prompt_template = prompt_templates.get(
        "generate_cypher_query_template_v5_with_tools", generate_cypher_query_template_v5_with_tools
    )
    prompt = prompt_template.format(filtered_graph=filtered_graph)

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
