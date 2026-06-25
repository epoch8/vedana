import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any, Protocol
from collections import defaultdict

from loguru import logger
from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from stell_core.configs import IntentDefinition, IntentsConfig
from stell_core.models import BotUtterEventDTO, TrackerEventDTO, UserInputEventDTO


def build_system_entities_extractor_prompt(entities: list[str]) -> str:
    entities_block = "\n".join(f"- {e}" for e in entities)
    entities_list = ", ".join(entities)

    return f"""Ты - точный экстрактор именованных сущностей.
Извлекай из сообщения пользователя значения для указанных типов сущностей.

ТИПЫ СУЩНОСТЕЙ КОТОРЫЕ НУЖНО ИЗВЛЕЧЬ:
{entities_block}

ИНСТРУКЦИЯ:
1. Если предоставлена история диалога — используй её для понимания контекста (какой вопрос задал бот).
2. Найди в сообщении пользователя значения для сущностей из списка выше.
3. Если бот явно спросил о конкретной сущности (например, про GitHub, базы данных, опыт),
   а пользователь ответил — считай ответ значением этой сущности, даже если ответ отрицательный
   (нет, неа, не знаю, прочерк и т.п.).
4. Если сущность не найдена и контекст не помогает — не включай её в результат.
5. Верни ТОЛЬКО JSON с найденными сущностями.

ВАЖНО:
- НЕ добавляй пояснения
- ТОЛЬКО JSON формат
- Ключи только из: {entities_list}
- Если ничего не найдено — верни {{}}"""


SYSTEM_INTENTS = {"nlu_fallback", "fallback"}


def _is_placeholder_intent(intent: IntentDefinition) -> bool:
    if intent.name in SYSTEM_INTENTS:
        return True
    return bool(intent.examples) and all(
        ex.rstrip("!. ").strip() == intent.name for ex in intent.examples
    )


def _intent_category(intent_name: str) -> str:
    return intent_name.split(".")[0] if "." in intent_name else intent_name


def build_system_intents_prompt(intents_config: IntentsConfig) -> str:
    active_intents = [i for i in intents_config.intents if not _is_placeholder_intent(i)]
    n = len(active_intents)

    categories: dict[str, list[tuple[int, Any]]] = defaultdict(list)
    for i, intent in enumerate(active_intents, start=1):
        categories[_intent_category(intent.name)].append((i, intent))

    blocks: list[str] = []
    for cat in sorted(categories):
        cat_lines = [f"  [{cat.upper()}]"]
        for i, intent in categories[cat]:
            unique_exs = list(dict.fromkeys([intent.description] + intent.examples))[:4]
            examples_str = " | ".join(f'"{ex}"' for ex in unique_exs)
            cat_lines.append(f"  {i}. {examples_str}")
        blocks.append("\n".join(cat_lines))

    intents_block = "\n\n".join(blocks)

    return f"""Ты — классификатор намерений. Намерения пронумерованы от 1 до {n}.

НАМЕРЕНИЯ (сгруппированы по теме; формат: номер. пример1 | пример2 | ...):
{intents_block}

ИНСТРУКЦИЯ:
1. Определи тему сообщения — загляни в нужную группу намерений.
2. Выбери намерение, примеры которого максимально близки по смыслу. Учитывай синонимы, разговорный стиль, опечатки.
3. История диалога (если есть) помогает понять короткие фразы: "да", "нет", "хочу".
4. Если несколько интентов похожи — выбирай наиболее специфичный по контексту.
5. Верни {{"id": 0}} ТОЛЬКО если:
   - сообщение явно не относится к тематике магазина (рецепты, здоровье, погода, политика и т.п.)
   - сообщение совершенно бессмысленно или нераспознаваемо
   Если тема понятна, но ты не уверен между несколькими вариантами — всё равно выбирай лучший, не возвращай 0.
6. Верни ТОЛЬКО JSON: {{"id": <число от 0 до {n}>}}

ТОЛЬКО JSON, никакого текста."""


def build_nlu_history(
    events: list[TrackerEventDTO],
    max_messages: int = 10,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for event in events:
        if isinstance(event, UserInputEventDTO):
            messages.append({"role": "user", "text": event.text})
        elif isinstance(event, BotUtterEventDTO):
            messages.append({"role": "bot", "text": event.text})
    return messages[-max_messages:]


class IntentClassifierProtocol(Protocol):
    async def classify(self, user_message: str, history: list[dict[str, str]] | None = None) -> str: ...


class EntitiiesExtractorProtocol(Protocol):
    async def classify(self, user_message: str) -> dict[str, Any]: ...


class IntentClassifier:
    def __init__(
        self,
        intents_config: IntentsConfig,
        model: str,
        extra_body: dict | None = None,
    ):
        self.intents_config = intents_config
        active = [i for i in intents_config.intents if not _is_placeholder_intent(i)]
        self.available_intents: list[str] = [intent.name for intent in active]
        self.system_prompt = build_system_intents_prompt(intents_config)
        self.agent = Agent(model=model, system_prompt=self.system_prompt)
        self._extra_body = extra_body

    async def classify(self, user_message: str, history: list[dict[str, str]] | None = None) -> str:
        model_settings = ModelSettings(
            max_tokens=50,
            timeout=120.0,
            temperature=0.0,
            extra_body=self._extra_body,
        )
        if history:
            lines = [f"{'Пользователь' if m['role'] == 'user' else 'Бот'}: {m['text']}" for m in history]
            history_block = "ИСТОРИЯ:\n" + "\n".join(lines) + "\n\n"
            user_prompt = f'{history_block}Сообщение: "{user_message}"'
        else:
            user_prompt = f'Сообщение: "{user_message}"'

        result = await self.agent.run(user_prompt, model_settings=model_settings)

        raw_output = result.output.strip()
        raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        logger.debug(f"Raw model output: {raw_output!r}")

        if "```" in raw_output:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
            if match:
                raw_output = match.group(1)

        id_match = re.search(r'"id"\s*:\s*(\d+)', raw_output)
        if id_match:
            intent_id = int(id_match.group(1))
            logger.debug(f"Extracted intent id: {intent_id}")
            if 1 <= intent_id <= len(self.available_intents):
                return self.available_intents[intent_id - 1]
            return "nlu_fallback"

        try:
            data = json.loads(raw_output)
            if isinstance(data, dict):
                raw_id = data.get("id")
                if isinstance(raw_id, int) and 1 <= raw_id <= len(self.available_intents):
                    return self.available_intents[raw_id - 1]
        except (json.JSONDecodeError, ValueError) as e:
            logger.debug(f"JSON parse failed: {e}")

        logger.debug(f"Could not parse intent id from: {raw_output!r}, falling back")
        return "nlu_fallback"


class LLMEnititiesExtractor:
    def __init__(
        self,
        entities: list[str],
        model: str,
        extra_body: dict | None = None,
    ) -> None:
        self.entities = entities
        self.system_prompt = build_system_entities_extractor_prompt(entities)
        self.agent = Agent(
            model=model,
            system_prompt=self.system_prompt,
        )
        self._extra_body = extra_body

    async def classify(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        active_slot: str | None = None,
    ) -> dict[str, Any]:
        model_settings = ModelSettings(
            max_tokens=150,
            timeout=120.0,
            temperature=0.0,
            extra_body=self._extra_body,
        )
        parts: list[str] = []
        if history:
            lines = [f"{'Пользователь' if m['role'] == 'user' else 'Бот'}: {m['text']}" for m in history]
            parts.append("ИСТОРИЯ:\n" + "\n".join(lines))
        if active_slot:
            parts.append(f"СЕЙЧАС СОБИРАЕТСЯ СЛОТ: {active_slot}")
        parts.append(f"Сообщение пользователя: '{user_message}'")
        user_prompt = "\n\n".join(parts)

        result = await self.agent.run(
            user_prompt,
            model_settings=model_settings,
        )
        raw_output = result.output.strip()
        raw_output = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL).strip()
        logger.debug(f"Raw entities output: {raw_output!r}")

        start = raw_output.find("{")
        end = raw_output.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                parsed = json.loads(raw_output[start : end + 1])
                if isinstance(parsed, dict):
                   return {k: v for k, v in parsed.items() if k in self.entities}
            except (json.JSONDecodeError, ValueError):
                pass

        logger.debug("Entities parse failed, returning empty dict")
        return {}


async def intent_classify(
    intent_classifier: IntentClassifierProtocol,
    text: str,
    history: list[dict[str, str]] | None = None,
) -> str:
    return await intent_classifier.classify(user_message=text, history=history)


async def llm_entities_extract(
    entity_extractor: LLMEnititiesExtractor,
    text: str,
    history: list[dict[str, str]] | None = None,
    active_slot: str | None = None,
) -> dict[str, Any]:
    return await entity_extractor.classify(user_message=text, history=history, active_slot=active_slot)


class RegexEntitiesExtractor:
    def __init__(
        self,
        entities: list[str],
        patterns: dict[str, list[str]],
    ) -> None:
        self.entities = entities
        self._compiled: dict[str, list[re.Pattern[str]]] = {
            entity: [re.compile(p, re.IGNORECASE) for p in patterns.get(entity, [])] for entity in entities
        }

    @property
    def available_entities(self) -> list[str]:
        return self.entities

    async def classify(self, user_message: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for entity, compiled in self._compiled.items():
            for pattern in compiled:
                m = pattern.search(user_message)
                if not m:
                    continue
                result[entity] = m.group(1).strip() if m.lastindex else m.group(0).strip()
                break
        return result


async def regex_entities_extract(
    extractor: RegexEntitiesExtractor,
    text: str,
) -> dict[str, Any]:
    return await extractor.classify(user_message=text)


@dataclass
class NLUResult:
    intent: str
    entities: dict[str, Any]
    user_input_text: str
    confidence: float | None = None


async def nlu_pipeline_runner(
    text: str,
    intent_classifier: IntentClassifierProtocol,
    entity_extractor: LLMEnititiesExtractor,
    regex_extractor: RegexEntitiesExtractor | None = None,
    history: list[dict[str, str]] | None = None,
    active_slot: str | None = None,
) -> NLUResult:
    intent_task = asyncio.create_task(intent_classify(text=text, intent_classifier=intent_classifier, history=history))
    entities_task = asyncio.create_task(
        llm_entities_extract(text=text, entity_extractor=entity_extractor, history=history, active_slot=active_slot)
    )
    intent = await intent_task
    entities = await entities_task
    if not entities and regex_extractor is not None:
        entities = await regex_entities_extract(extractor=regex_extractor, text=text)
    return NLUResult(intent=intent, entities=entities, user_input_text=text)
