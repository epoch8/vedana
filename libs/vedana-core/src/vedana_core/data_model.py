import asyncio
import json
from dataclasses import dataclass
from typing import Any, cast

import pandas as pd
import sqlalchemy.ext.asyncio as sa_aio

from config_plane.impl.sql import create_sql_config_repo
from vedana_core.db import get_config_plane_sessionmaker
from vedana_core.settings import settings as core_settings

@dataclass
class Attribute:
    name: str
    description: str
    example: str
    dtype: str
    query: str
    embeddable: bool = False
    embed_threshold: float = 0


@dataclass
class Anchor:
    noun: str
    description: str
    id_example: str
    query: str
    attributes: list[Attribute]

    def __str__(self) -> str:
        return self.noun


@dataclass
class Link:
    anchor_from: Anchor
    anchor_to: Anchor
    sentence: str
    description: str
    query: str
    attributes: list[Attribute]
    has_direction: bool = False
    anchor_from_link_attr_name: str = ""
    anchor_to_link_attr_name: str = ""


@dataclass
class Query:
    name: str
    example: str


@dataclass
class ConversationLifecycleEvent:
    event: str
    text: str


@dataclass
class Prompt:
    name: str
    text: str


class DataModel:
    """
    DataModel, loads from config-plane only.
    """

    def __init__(
        self,
        sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession],
        config_plane_branch: str | None = None,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.config_plane_branch = config_plane_branch or core_settings.config_plane_branch
        self._config_snapshot_override: int | None = None
        self._config_cache_snapshot_id: int | None = None
        self._config_cache_payload: dict[str, Any] | None = None
        self._config_cache_parsed: dict[str, Any] | None = None
        self._config_repo = create_sql_config_repo(get_config_plane_sessionmaker(), branch=self.config_plane_branch)

    @classmethod
    def create(cls, sessionmaker) -> "DataModel":
        return cls(sessionmaker=sessionmaker)

    def set_branch(self, branch: str) -> None:
        self.config_plane_branch = branch
        self._config_repo.switch_branch(branch)
        self._config_snapshot_override = None
        self._config_cache_snapshot_id = None
        self._config_cache_payload = None
        self._config_cache_parsed = None

    def set_snapshot_override(self, snapshot_id: int | None) -> None:
        self._config_snapshot_override = snapshot_id
        # self._config_repo.set_branch_snapshot_id(snapshot_id=snapshot_id, branch=self.config_plane_branch)
        self._config_cache_snapshot_id = None
        self._config_cache_payload = None
        self._config_cache_parsed = None

    def get_snapshot_id(self) -> int | None:
        if self._config_snapshot_override is not None:
            return self._config_snapshot_override
        snapshot_id = self._config_repo.get_branch_snapshot_id(self.config_plane_branch)
        if snapshot_id is None:
            return None
        try:
            return int(snapshot_id)
        except ValueError:
            return None

    async def _get_config_payload(self) -> dict[str, Any]:
        snapshot_id = self._config_snapshot_override
        if snapshot_id is None:
            snapshot_id_str = await asyncio.to_thread(
                self._config_repo.get_branch_snapshot_id, self.config_plane_branch
            )
            if snapshot_id_str is not None:
                try:
                    snapshot_id = int(snapshot_id_str)
                except ValueError:
                    snapshot_id = None
        if snapshot_id is None:
            raise RuntimeError(f"No committed config-plane snapshot for branch '{self.config_plane_branch}'")

        if (
            self._config_cache_snapshot_id == snapshot_id
            and self._config_cache_payload is not None
        ):
            return self._config_cache_payload

        raw = await asyncio.to_thread(
            self._config_repo.get, "vedana.data_model", snapshot_id=str(snapshot_id)
        )
        if raw is None:
            exists = await asyncio.to_thread(
                self._config_repo.snapshot_exists, str(snapshot_id)
            )
            if not exists:
                raise RuntimeError(f"Data model snapshot ID {snapshot_id} does not exist")
            raise RuntimeError(f"Data model snapshot ID {snapshot_id} is missing key 'vedana.data_model'")

        payload = json.loads(raw.decode("utf-8"))
        self._config_cache_snapshot_id = snapshot_id
        self._config_cache_payload = payload
        self._config_cache_parsed = None
        return payload

    async def _get_config_parsed(self) -> dict[str, Any]:
        payload = await self._get_config_payload()

        if self._config_cache_parsed is not None:
            return self._config_cache_parsed

        anchors: dict[str, Anchor] = {}
        for a in payload.get("anchors", []) or []:
            noun = str(a.get("noun", "")).strip()
            if not noun:
                continue
            anchors[noun] = Anchor(
                noun=noun,
                description=str(a.get("description", "")),
                id_example=str(a.get("id_example", "")),
                query=str(a.get("query", "")),
                attributes=[],
            )

        for a in payload.get("anchors", []) or []:
            noun = str(a.get("noun", "")).strip()
            if noun not in anchors:
                continue
            for attr in a.get("attributes", []) or []:
                anchors[noun].attributes.append(
                    Attribute(
                        name=str(attr.get("attribute_name", "")),
                        description=str(attr.get("description", "")),
                        example=str(attr.get("data_example", "")),
                        dtype=str(attr.get("dtype", "")),
                        query=str(attr.get("query", "")),
                        embeddable=bool(attr.get("embeddable", False)),
                        embed_threshold=float(attr.get("embed_threshold", 1.0)),
                    )
                )

        links: dict[str, Link] = {}
        for li in payload.get("links", []) or []:
            sentence = str(li.get("sentence", "")).strip()
            if not sentence:
                continue
            anchor_from = anchors.get(str(li.get("anchor1", "")).strip())
            anchor_to = anchors.get(str(li.get("anchor2", "")).strip())
            if anchor_from is None or anchor_to is None:
                continue
            links[sentence] = Link(
                anchor_from=anchor_from,
                anchor_to=anchor_to,
                sentence=sentence,
                description=str(li.get("description", "")),
                query=str(li.get("query", "")),
                attributes=[],
                has_direction=bool(li.get("has_direction", False)),
                anchor_from_link_attr_name=str(li.get("anchor1_link_column_name", "")),
                anchor_to_link_attr_name=str(li.get("anchor2_link_column_name", "")),
            )

        for li in payload.get("links", []) or []:
            sentence = str(li.get("sentence", "")).strip()
            if sentence not in links:
                continue
            for attr in li.get("attributes", []) or []:
                links[sentence].attributes.append(
                    Attribute(
                        name=str(attr.get("attribute_name", "")),
                        description=str(attr.get("description", "")),
                        example=str(attr.get("data_example", "")),
                        dtype=str(attr.get("dtype", "")),
                        query=str(attr.get("query", "")),
                        embeddable=bool(attr.get("embeddable", False)),
                        embed_threshold=float(attr.get("embed_threshold", 1.0)),
                    )
                )

        queries = [
            Query(name=str(q.get("name", "")), example=str(q.get("example", "")))
            for q in (payload.get("queries", []) or [])
            if str(q.get("name", "")).strip()
        ]

        prompts = [
            Prompt(name=str(p.get("name", "")), text=str(p.get("text", "")))
            for p in (payload.get("prompts", []) or [])
            if str(p.get("name", "")).strip()
        ]

        lifecycle = [
            ConversationLifecycleEvent(event=str(e.get("event", "")), text=str(e.get("text", "")))
            for e in (payload.get("conversation_lifecycle", []) or [])
            if str(e.get("event", "")).strip()
        ]

        self._config_cache_parsed = {
            "anchors": list(anchors.values()),
            "links": list(links.values()),
            "queries": queries,
            "prompts": prompts,
            "conversation_lifecycle": lifecycle,
        }
        return self._config_cache_parsed

    def update_from_grist(self, branch: str | None = None) -> int | None:
        payload = _build_payload_from_grist()
        blob = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        target_branch = branch or core_settings.config_plane_dev_branch
        current_branch = self.config_plane_branch
        if target_branch != current_branch:
            self._config_repo.switch_branch(target_branch)
        self._config_repo.set("vedana.data_model", blob)
        self._config_repo.commit()
        snapshot_id = self._config_repo.get_branch_snapshot_id(target_branch)
        if target_branch != current_branch:
            self._config_repo.switch_branch(current_branch)
        return int(snapshot_id) if snapshot_id is not None else None

    async def get_anchors(self) -> list[Anchor]:
        parsed = await self._get_config_parsed()
        return parsed["anchors"]

    async def get_links(self, anchors_dict: dict[str, Anchor] | None = None) -> list[Link]:
        parsed = await self._get_config_parsed()
        if anchors_dict is None:
            return parsed["links"]
        remapped: list[Link] = []
        for link in parsed["links"]:
            anchor_from = anchors_dict.get(link.anchor_from.noun)
            anchor_to = anchors_dict.get(link.anchor_to.noun)
            if anchor_from is None or anchor_to is None:
                continue
            remapped.append(
                Link(
                    anchor_from=anchor_from,
                    anchor_to=anchor_to,
                    sentence=link.sentence,
                    description=link.description,
                    query=link.query,
                    attributes=link.attributes,
                    has_direction=link.has_direction,
                    anchor_from_link_attr_name=link.anchor_from_link_attr_name,
                    anchor_to_link_attr_name=link.anchor_to_link_attr_name,
                )
            )
        return remapped

    async def get_queries(self) -> list[Query]:
        parsed = await self._get_config_parsed()  # todo simplify
        return parsed["queries"]

    async def get_conversation_lifecycle_events(self) -> list[ConversationLifecycleEvent]:
        parsed = await self._get_config_parsed()  # todo simplify
        return parsed["conversation_lifecycle"]

    async def conversation_lifecycle_events(self) -> dict[str, str]:
        cl = await self.get_conversation_lifecycle_events()
        return {c.event: c.text for c in cl}

    async def get_prompts(self) -> list[Prompt]:
        parsed = await self._get_config_parsed()  # todo simplify
        return parsed["prompts"]

    async def prompt_templates(self) -> dict[str, str]:
        prompts = await self.get_prompts()
        return {p.name: p.text for p in prompts}

    async def vector_indices(self) -> list[tuple[str, str, str, float]]:
        """
        returns list
        ("anchor", anchor.noun, anchor.attribute, anchor.th) +
        ("edge", link.sentence, link.attribute, link.th)
        for all embeddable attributes
        """
        anchors = await self.get_anchors()
        links = await self.get_links(anchors_dict={a.noun: a for a in anchors})

        a_i = [
            ("anchor", anchor.noun, attr.name, attr.embed_threshold)
            for anchor in anchors
            for attr in anchor.attributes
            if attr.embeddable
        ]
        l_i = [
            ("edge", link.sentence, attr.name, attr.embed_threshold)
            for link in links
            for attr in link.attributes
            if attr.embeddable
        ]
        return a_i + l_i

    async def anchor_links(self, anchor_noun: str) -> list[Link]:
        """all links that connect to/from this anchor"""
        links = await self.get_links()
        return [
            link
            for link in links
            if (link.anchor_from.noun == anchor_noun and link.anchor_from_link_attr_name)
            or (link.anchor_to.noun == anchor_noun and link.anchor_to_link_attr_name)
        ]

    async def to_text_descr(
        self,
        anchor_nouns: list[str] | None = None,
        link_sentences: list[str] | None = None,
        anchor_attribute_names: list[str] | None = None,
        link_attribute_names: list[str] | None = None,
        query_names: list[str] | None = None,
    ) -> str:
        """Create a text description of the data model, optionally filtered.

        Args:
            anchor_nouns: List of anchor nouns to include. If None, includes all.
            link_sentences: List of link sentences to include. If None, includes all.
            anchor_attribute_names: List of anchor attribute names to include. If None, includes all.
            link_attribute_names: List of link attribute names to include. If None, includes all.
            query_names: List of query names to include. If None, includes all.

        Returns:
            A formatted string description of the data model.
        """
        anchors = await self.get_anchors()
        links = await self.get_links(anchors_dict={a.noun: a for a in anchors})
        queries = await self.get_queries()
        dm_templates = await self.prompt_templates()

        # Convert to sets for efficient lookup, None means include all
        anchor_set = set(anchor_nouns) if anchor_nouns is not None else None
        link_set = set(link_sentences) if link_sentences is not None else None
        anchor_attr_set = set(anchor_attribute_names) if anchor_attribute_names is not None else None
        link_attr_set = set(link_attribute_names) if link_attribute_names is not None else None
        query_set = set(query_names) if query_names is not None else None

        # Filter anchors
        filtered_anchors = [anchor for anchor in anchors if anchor_set is None or anchor.noun in anchor_set]

        # Create a map for quick anchor lookup (for link filtering)
        anchors_map = {a.noun: a for a in filtered_anchors}

        # Filter links (only include if both anchors are in filtered set)
        filtered_links = [
            link
            for link in links
            if (link_set is None or link.sentence in link_set)
            and link.anchor_from.noun in anchors_map
            and link.anchor_to.noun in anchors_map
        ]

        anchor_descr = "\n".join(
            dm_templates.get("dm_anchor_descr_template", dm_anchor_descr_template).format(anchor=anchor)
            for anchor in filtered_anchors
        )

        anchor_attrs_descr = "\n".join(
            dm_templates.get("dm_attr_descr_template", dm_attr_descr_template).format(anchor=anchor, attr=attr)
            for anchor in filtered_anchors
            for attr in anchor.attributes
            if anchor_attr_set is None or attr.name in anchor_attr_set
        )

        link_descr = "\n".join(
            dm_templates.get("dm_link_descr_template", dm_link_descr_template).format(link=link)
            for link in filtered_links
        )

        link_attrs_descr = "\n".join(
            dm_templates.get("dm_link_attr_descr_template", dm_link_attr_descr_template).format(link=link, attr=attr)
            for link in filtered_links
            for attr in link.attributes
            if link_attr_set is None or attr.name in link_attr_set
        )

        filtered_queries = [q for q in queries if query_set is None or q.name in query_set]
        queries_descr = "\n".join(
            dm_templates.get("dm_query_descr_template", dm_query_descr_template).format(query=query)
            for query in filtered_queries
        )

        dm_template = dm_templates.get("dm_descr_template", dm_descr_template)

        return dm_template.format(
            anchors=anchor_descr,
            anchor_attrs=anchor_attrs_descr,
            links=link_descr,
            link_attrs=link_attrs_descr,
            queries=queries_descr,
        )

    async def to_compact_json(self) -> dict:
        anchors = await self.get_anchors()
        links = await self.get_links(anchors_dict={a.noun: a for a in anchors})
        queries = await self.get_queries()

        descr = {
            "anchors": [
                {
                    "name": a.noun,
                    "description": a.description,
                    "example": a.id_example,
                    "attributes": [
                        {
                            "attr_name": aa.name,
                            "attr_description": aa.description,
                        }
                        for aa in a.attributes
                    ],
                }
                for a in anchors
            ],
            "links": [
                {
                    "from": li.anchor_from,
                    "to": li.anchor_to,
                    "sentence": li.sentence,
                    "description": li.description,
                    "attributes": [
                        {
                            "attr_name": la.name,
                            "attr_description": la.description,
                        }
                        for la in li.attributes
                    ],
                }
                for li in links
            ],
            "queries": {i: q.name for i, q in enumerate(queries, start=1)},
        }
        return descr


# default templates
dm_descr_template = """\
## Узлы:
{anchors}

## Атрибуты узлов:
{anchor_attrs}

## Связи между узлами:
{links}

## Атрибуты связей:
{link_attrs}

## Типичные вопросы:
{queries}
"""

dm_anchor_descr_template = (
    "- {anchor.noun}: {anchor.description}; пример ID: {anchor.id_example}; запрос для получения: {anchor.query}"
)
dm_attr_descr_template = (
    "- {anchor.noun}.{attr.name}: {attr.description}; пример: {attr.example}; запрос для получения: {attr.query}"
)
dm_link_descr_template = "- {link.sentence}: {link.description}; пример запроса: {link.query}"
dm_link_attr_descr_template = (
    "- {link.sentence}.{attr.name}: {attr.description}; пример: {attr.example}; запрос для получения: {attr.query}"
)
dm_query_descr_template = "- {query.name}\n{query.example}"

# Compact templates (without cypher queries)
dm_compact_descr_template = """\
## Узлы:
{anchors}

## Атрибуты узлов:
{anchor_attrs}

## Связи между узлами:
{links}

## Атрибуты связей:
{link_attrs}

## Сценарии вопросов:
{queries}
"""

dm_compact_anchor_descr_template = "- {anchor.noun}: {anchor.description}"
dm_compact_attr_descr_template = "- {anchor.noun}.{attr.name}: {attr.description}"
dm_compact_link_descr_template = "- {link.sentence}: {link.description}"
dm_compact_link_attr_descr_template = "- {link.sentence}.{attr.name}: {attr.description}"
dm_compact_query_descr_template = "- {query.name}"


def _build_payload_from_grist() -> dict[str, Any]:
    from vedana_core.data_provider import GristCsvDataProvider  # todo mv

    loader = GristCsvDataProvider(
        doc_id=core_settings.grist_data_model_doc_id,
        grist_server=core_settings.grist_server_url,
        api_key=core_settings.grist_api_key,
    )

    links_df = cast(
        pd.DataFrame,
        loader.get_table("Links")[
            [
                "anchor1",
                "anchor2",
                "sentence",
                "description",
                "query",
                "anchor1_link_column_name",
                "anchor2_link_column_name",
                "has_direction",
            ]
        ],
    )
    links_df["has_direction"] = links_df["has_direction"].astype(bool)
    links_df = links_df.dropna(subset=["anchor1", "anchor2", "sentence"], inplace=False)

    anchor_attrs_df = cast(
        pd.DataFrame,
        loader.get_table("Anchor_attributes")[
            [
                "anchor",
                "attribute_name",
                "description",
                "data_example",
                "embeddable",
                "query",
                "dtype",
                "embed_threshold",
            ]
        ],
    )
    anchor_attrs_df["embeddable"] = anchor_attrs_df["embeddable"].astype(bool)
    anchor_attrs_df["embed_threshold"] = anchor_attrs_df["embed_threshold"].astype(float)
    anchor_attrs_df = anchor_attrs_df.dropna(subset=["anchor", "attribute_name"], how="any")

    link_attrs_df = cast(
        pd.DataFrame,
        loader.get_table("Link_attributes")[
            [
                "link",
                "attribute_name",
                "description",
                "data_example",
                "embeddable",
                "query",
                "dtype",
                "embed_threshold",
            ]
        ],
    )
    link_attrs_df["embeddable"] = link_attrs_df["embeddable"].astype(bool)
    link_attrs_df["embed_threshold"] = link_attrs_df["embed_threshold"].astype(float)
    link_attrs_df = link_attrs_df.dropna(subset=["link", "attribute_name"], how="any")

    anchors_df = cast(
        pd.DataFrame,
        loader.get_table("Anchors")[
            [
                "noun",
                "description",
                "id_example",
                "query",
            ]
        ],
    )
    anchors_df = anchors_df.dropna(subset=["noun"], inplace=False)
    anchors_df = anchors_df.astype(str)

    queries_df = cast(pd.DataFrame, loader.get_table("Queries")[["query_name", "query_example"]])
    queries_df = queries_df.dropna().astype(str)

    prompts_df = cast(pd.DataFrame, loader.get_table("Prompts")[["name", "text"]])
    prompts_df = prompts_df.dropna().astype(str)

    conversation_lifecycle_df = cast(
        pd.DataFrame,
        loader.get_table("ConversationLifecycle")[["event", "text"]],
    )
    conversation_lifecycle_df = conversation_lifecycle_df.dropna().astype(str)

    anchors = anchors_df.astype(object).where(pd.notna(anchors_df), None).to_dict(orient="records")
    anchor_attrs = anchor_attrs_df.astype(object).where(pd.notna(anchor_attrs_df), None).to_dict(orient="records")
    links = links_df.astype(object).where(pd.notna(links_df), None).to_dict(orient="records")
    link_attrs = link_attrs_df.astype(object).where(pd.notna(link_attrs_df), None).to_dict(orient="records")
    queries = queries_df.astype(object).where(pd.notna(queries_df), None).to_dict(orient="records")
    prompts = prompts_df.astype(object).where(pd.notna(prompts_df), None).to_dict(orient="records")
    lifecycle = (
        conversation_lifecycle_df.astype(object)
        .where(pd.notna(conversation_lifecycle_df), None)
        .to_dict(orient="records")
    )

    anchors_by: dict[str, dict[str, Any]] = {}
    for a in anchors:
        noun = str(a.get("noun", "")).strip()
        if not noun:
            continue
        anchors_by[noun] = {
            "noun": noun,
            "description": a.get("description", ""),
            "id_example": a.get("id_example", ""),
            "query": a.get("query", ""),
            "attributes": [],
        }

    for attr in anchor_attrs:
        noun = str(attr.get("anchor", "")).strip()
        if not noun or noun not in anchors_by:
            continue
        anchors_by[noun]["attributes"].append(
            {
                "attribute_name": attr.get("attribute_name", ""),
                "description": attr.get("description", ""),
                "data_example": attr.get("data_example", ""),
                "embeddable": bool(attr.get("embeddable", False)),
                "query": attr.get("query", ""),
                "dtype": attr.get("dtype", ""),
                "embed_threshold": float(attr.get("embed_threshold") or 1.0),
            }
        )

    links_by: dict[str, dict[str, Any]] = {}
    for li in links:
        sentence = str(li.get("sentence", "")).strip()
        if not sentence:
            continue
        links_by[sentence] = {
            "anchor1": li.get("anchor1", ""),
            "anchor2": li.get("anchor2", ""),
            "sentence": sentence,
            "description": li.get("description", ""),
            "query": li.get("query", ""),
            "anchor1_link_column_name": li.get("anchor1_link_column_name", ""),
            "anchor2_link_column_name": li.get("anchor2_link_column_name", ""),
            "has_direction": bool(li.get("has_direction", False)),
            "attributes": [],
        }

    for attr in link_attrs:
        sentence = str(attr.get("link", "")).strip()
        if not sentence or sentence not in links_by:
            continue
        links_by[sentence]["attributes"].append(
            {
                "attribute_name": attr.get("attribute_name", ""),
                "description": attr.get("description", ""),
                "data_example": attr.get("data_example", ""),
                "embeddable": bool(attr.get("embeddable", False)),
                "query": attr.get("query", ""),
                "dtype": attr.get("dtype", ""),
                "embed_threshold": float(attr.get("embed_threshold") or 1.0),
            }
        )

    return {
        "anchors": list(anchors_by.values()),
        "links": list(links_by.values()),
        "queries": [
            {"name": q.get("query_name", ""), "example": q.get("query_example", "")} for q in queries if q.get("query_name")
        ],
        "prompts": [{"name": p.get("name", ""), "text": p.get("text", "")} for p in prompts if p.get("name")],
        "conversation_lifecycle": [
            {"event": c.get("event", ""), "text": c.get("text", "")} for c in lifecycle if c.get("event")
        ],
    }
