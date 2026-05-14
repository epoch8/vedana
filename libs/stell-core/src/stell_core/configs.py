import dataclasses
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Self

import yaml
from loguru import logger
from pydantic import BaseModel, DirectoryPath, Field, FilePath, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class IntentDefinition(BaseModel):
    name: str
    description: str
    examples: list[str]


class IntentsConfig(BaseModel):
    intents: list[IntentDefinition]

    @classmethod
    def from_rasa_nlu_json(cls, path: Path) -> "IntentsConfig":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)

        examples: list[dict] = data["rasa_nlu_data"]["common_examples"]

        by_intent: dict[str, list[str]] = defaultdict(list)
        canonical: dict[str, str] = {}
        for ex in examples:
            intent = ex.get("intent", "")
            text = ex.get("text", "").strip()
            if not intent or not text:
                continue
            if ex.get("metadata", {}).get("canonical") and intent not in canonical:
                canonical[intent] = text
            by_intent[intent].append(text)

        intents: list[IntentDefinition] = []
        for intent_name, intent_examples in by_intent.items():
            canon = canonical.get(intent_name)
            description = canon or intent_examples[0]
            if canon and intent_examples and intent_examples[0] != canon:
                ordered = [canon] + [e for e in intent_examples if e != canon]
            else:
                ordered = intent_examples
            intents.append(
                IntentDefinition(
                    name=intent_name,
                    description=description,
                    examples=ordered,
                )
            )

        logger.info(f"Rasa NLU loaded from {path}: {len(intents)} intents, {len(examples)} examples")
        return cls(intents=intents)


class IntentStep(BaseModel):
    intent: str
    entities: list[str] = Field(default_factory=list)


class SlotWasSetStep(BaseModel):
    slot_was_set: dict[str, Any]

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        if isinstance(data, dict) and "slot_was_set" in data:
            raw = data["slot_was_set"]
            slots: dict[str, Any] = {}
            for item in raw:
                if isinstance(item, str):
                    slots[item] = dataclasses.MISSING
                elif isinstance(item, dict):
                    slots.update(item)
            return {"slot_was_set": slots}
        return data


class ActionStep(BaseModel):
    action: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "action" not in data:
            return data

        raw_action = data.get("action")
        name: str | None
        raw_params: Any | None

        if isinstance(raw_action, dict):
            name = raw_action.get("name")
            raw_params = raw_action.get("params")
        else:
            name = raw_action
            raw_params = data.get("params")

        params: dict[str, Any] = {}
        if isinstance(raw_params, dict):
            params = raw_params
        elif isinstance(raw_params, list):
            for item in raw_params:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    key, value = item
                    params[str(key)] = value

        return {"action": name, "params": params}


class ActiveLoopStep(BaseModel):
    active_loop: str | None


StoryStep = IntentStep | SlotWasSetStep | ActionStep | ActiveLoopStep


class StoryModel(BaseModel):
    story: str
    steps: list[StoryStep]
    metadata: dict[str, Any] = Field(default_factory=dict)


_SKIP_ACTIONS: set[str] = {"action_system_message"}
_MAX_BRANCH_PATHS = 32
_MAX_STEPS = 40


def clean_name(raw: str) -> str:
    name = re.sub(r"[^\w\s.\-]", "", raw).strip()
    return name[:80] or "unnamed"


def action_name(action_value: object) -> str | None:
    if isinstance(action_value, str):
        return action_value
    if isinstance(action_value, dict) and "name" in action_value:
        return action_value["name"]
    return None


def action_params(action_value: object) -> list | None:
    if isinstance(action_value, dict) and "params" in action_value:
        return action_value["params"]
    return None


def or_intents(step: dict) -> list[dict]:
    result: list[dict] = []
    for item in step.get("or") or []:
        if isinstance(item, dict) and "intent" in item:
            dm: dict = {"intent": item["intent"]}
            if item.get("entities"):
                dm["entities"] = item["entities"]
            result.append(dm)
    return result


def intent_step(step: dict) -> dict | None:
    if "intent" not in step:
        return None
    dm: dict = {"intent": step["intent"]}
    entities = [e for e in (step.get("entities") or []) if e]
    if entities:
        dm["entities"] = entities
    return dm


def first_checkpoint(story: dict) -> str | None:
    steps = story.get("steps") or []
    if steps and isinstance(steps[0], dict) and "checkpoint" in steps[0]:
        return steps[0]["checkpoint"]
    return None


def build_cp_index(stories: list[dict]) -> dict[str, list[dict]]:
    idx: dict[str, list[dict]] = defaultdict(list)
    for s in stories:
        cp = first_checkpoint(s)
        if cp is not None:
            idx[cp].append(s)
    return idx


def flatten(
    raw_steps: list,
    cp_idx: dict[str, list[dict]],
    name_idx: dict[str, dict],
    visited: frozenset[str],
) -> list[list[dict]]:
    paths: list[list[dict]] = [[]]

    for step in raw_steps:
        if not isinstance(step, dict):
            continue

        if "checkpoint" in step:
            cp = step["checkpoint"]
            if cp in visited:
                continue

            if cp.endswith("__branches"):
                branches = cp_idx.get(cp, [])
                if not branches:
                    continue
                new_paths: list[list[dict]] = []
                for branch in branches:
                    tail = (branch.get("steps") or [])[1:]
                    expanded = flatten(tail, cp_idx, name_idx, visited | {cp})
                    for exp in expanded:
                        for p in paths:
                            new_paths.append(p + exp)
                        if len(new_paths) >= _MAX_BRANCH_PATHS:
                            break
                    if len(new_paths) >= _MAX_BRANCH_PATHS:
                        break
                paths = new_paths[:_MAX_BRANCH_PATHS]

            elif cp.startswith("link-to-"):
                target_name = cp[len("link-to-") :].split("/")[0]
                linked_story = name_idx.get(target_name)
                if linked_story is None:
                    continue
                tail = linked_story.get("steps") or []
                expanded = flatten(tail, cp_idx, name_idx, visited | {cp})
                if not expanded:
                    continue
                if len(expanded) == 1:
                    paths = [p + expanded[0] for p in paths]
                else:
                    new_paths: list[list[dict]] = []
                    for exp in expanded:
                        for p in paths:
                            new_paths.append(p + exp)
                            if len(new_paths) >= _MAX_BRANCH_PATHS:
                                break
                        if len(new_paths) >= _MAX_BRANCH_PATHS:
                            break
                    paths = new_paths[:_MAX_BRANCH_PATHS]

            else:
                linked = cp_idx.get(cp, [])
                if not linked:
                    continue
                tail = (linked[0].get("steps") or [])[1:]
                expanded = flatten(tail, cp_idx, name_idx, visited | {cp})
                inline = expanded[0] if expanded else []
                paths = [p + inline for p in paths]
            continue

        if "active_loop" in step:
            for p in paths:
                p.append({"active_loop": step["active_loop"]})
            continue

        if "or" in step:
            variants = or_intents(step)
            if not variants:
                continue
            paths = [[*p, v] for v in variants for p in paths]
            continue

        if "intent" in step:
            dm = intent_step(step)
            if dm:
                for p in paths:
                    p.append(dm)
            continue

        if "slot_was_set" in step:
            for p in paths:
                p.append({"slot_was_set": step["slot_was_set"]})
            continue

        if "action" in step:
            name = action_name(step["action"])
            if name and name not in _SKIP_ACTIONS:
                params = action_params(step["action"])
                for p in paths:
                    if params is not None:
                        p.append({"action": {"name": name, "params": params}})
                    else:
                        p.append({"action": name})
            continue

    return paths


def _is_valid(steps: list[dict]) -> bool:
    has_intent = any("intent" in s for s in steps)
    has_action = any("action" in s for s in steps)
    return has_intent and has_action


class StoriesConfig(BaseModel):
    stories: list[StoryModel]

    @classmethod
    def from_botfront_dir(cls, path: Path) -> "StoriesConfig":
        raw_stories: list[dict] = []
        for file_path in sorted(path.rglob("*.yml")):
            with file_path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if not isinstance(data, dict):
                continue
            src = file_path.relative_to(path).as_posix()
            for s in data.get("stories", []):
                if (s.get("metadata") or {}).get("status") == "unpublished":
                    continue
                s["_source"] = src
                raw_stories.append(s)
            for r in data.get("rules", []):
                if (r.get("metadata") or {}).get("status") == "unpublished":
                    continue
                rule_as_story = dict(r)
                rule_as_story["story"] = rule_as_story.pop("rule", "unnamed_rule")
                rule_as_story["_source"] = src
                raw_stories.append(rule_as_story)

        cp_idx = build_cp_index(raw_stories)
        name_idx = {s["story"]: s for s in raw_stories if s.get("story")}
        entries = [s for s in raw_stories if first_checkpoint(s) is None]

        story_models: list[StoryModel] = []
        seen: set[str] = set()
        counters: dict[str, int] = {"ok": 0, "no_intent_action": 0, "too_long": 0, "empty": 0}

        for story in entries:
            raw_steps = story.get("steps") or []
            group = (story.get("metadata") or {}).get("group", "")
            base = clean_name(story.get("story", "unnamed"))

            paths = flatten(raw_steps, cp_idx, name_idx, frozenset())

            for i, path_steps in enumerate(paths):
                if not path_steps:
                    counters["empty"] += 1
                    continue
                if len(path_steps) > _MAX_STEPS:
                    counters["too_long"] += 1
                    continue
                if not _is_valid(path_steps):
                    counters["no_intent_action"] += 1
                    continue

                name = base if len(paths) == 1 else f"{base}__{i}"
                if name in seen:
                    name = f"{name}_{len(seen)}"
                seen.add(name)

                metadata: dict[str, Any] = {}
                if group:
                    metadata["group"] = group

                story_models.append(
                    StoryModel.model_validate(
                        {
                            "story": name,
                            "steps": path_steps,
                            "metadata": metadata,
                        }
                    )
                )
                counters["ok"] += 1

        logger.info(
            f"Botfront stories loaded from {path}: "
            f"ok={counters['ok']} skipped(no_intent_action)={counters['no_intent_action']} "
            f"skipped(too_long)={counters['too_long']} skipped(empty)={counters['empty']}"
        )
        return cls(stories=story_models)


class YamlConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    stories_yaml_path: FilePath | None = None
    botfront_stories_dir: DirectoryPath | None = None
    domain_yaml_path: FilePath
    intents_yaml_path: FilePath | None = None
    nlu_json_path: FilePath | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    llm_provider: str = "ollama"
    llm_model: str = "llama3.2:3b"
    openai_api_key: str | None = None
    openai_base_url: str | None = None

    @model_validator(mode="after")
    def check_sources(self) -> Self:
        if self.stories_yaml_path is None and self.botfront_stories_dir is None:
            raise ValueError("Either STORIES_YAML_PATH or BOTFRONT_STORIES_DIR must be set in .env")
        if self.intents_yaml_path is None and self.nlu_json_path is None:
            raise ValueError("Either INTENTS_YAML_PATH or NLU_JSON_PATH must be set in .env")
        return self

    @property
    def effective_llm_model(self) -> str:
        return f"{self.llm_provider}:{self.llm_model}"

    @property
    def stories(self) -> StoriesConfig:
        if self.botfront_stories_dir is not None:
            return StoriesConfig.from_botfront_dir(self.botfront_stories_dir)
        with Path.open(self.stories_yaml_path) as file:  # type: ignore[arg-type]
            data = yaml.safe_load(file)
        return StoriesConfig.model_validate(data)

    @property
    def domain(self) -> dict[str, Any]:
        with Path.open(self.domain_yaml_path) as file:
            data = yaml.safe_load(file)

        actions: list[str] = []
        actions_params: dict[str, dict[str, Any]] = {}
        for entry in data.get("actions", []):
            if isinstance(entry, str):
                actions.append(entry)
            elif isinstance(entry, dict):
                name, params = next(iter(entry.items()))
                actions.append(name)
                actions_params[name] = params or {}

        return {
            "intents": data["intents"],
            "slots": data["slots"],
            "entities": data["entities"],
            "responses": data["responses"],
            "actions": actions,
            "actions_params": actions_params,
            "forms": data.get("forms", {}),
        }

    @property
    def intents(self) -> IntentsConfig:
        if self.nlu_json_path is not None:
            return IntentsConfig.from_rasa_nlu_json(self.nlu_json_path)
        with Path.open(self.intents_yaml_path) as file:  # type: ignore[arg-type]
            data = yaml.safe_load(file)
        if "rasa_nlu_data" in data:
            return IntentsConfig.from_rasa_nlu_json(self.intents_yaml_path)  # type: ignore[arg-type]
        return IntentsConfig.model_validate(data)


yaml_config = YamlConfig()  # type: ignore
os.environ.setdefault("OLLAMA_BASE_URL", yaml_config.ollama_base_url)
if yaml_config.openai_api_key:
    os.environ.setdefault("OPENAI_API_KEY", yaml_config.openai_api_key)
if yaml_config.openai_base_url:
    os.environ.setdefault("OPENAI_BASE_URL", yaml_config.openai_base_url)
logger.debug("YAML Config loaded successfully.")
