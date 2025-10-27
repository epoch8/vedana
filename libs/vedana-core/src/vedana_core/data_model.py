import abc
import json
import logging
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, cast

import grist_api
import pandas as pd
import requests
import requests.exceptions

from vedana_core.graph import Graph

logger = logging.getLogger(__name__)


@dataclass
class Attribute:
    name: str
    description: str
    example: str
    dtype: str
    query: str
    meta: dict[str, Any]
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


class DataModel:  # TODO refer to SQL only for all operations. All ops should get replaced with database read queries.
    def __init__(
        self,
        anchors: list[Anchor],
        links: list[Link],
        attrs: list[Attribute],
        queries: list[Query],
        conversation_lifecycle: list[ConversationLifecycleEvent],
        prompts: list[Prompt],
    ) -> None:
        self.anchors = anchors
        self.links = links
        self.attrs = attrs
        self.queries = queries
        self.conversation_lifecycle = conversation_lifecycle
        self.prompts = prompts

    @classmethod
    def _load(cls, loader: DataModelLoader) -> "DataModel":
        # TODO relpace with reads during func calls
        anchors = {
            noun: Anchor(
                noun=noun,
                description=description,
                id_example=id_example,
                query=query,
                attributes=[],
            )
            for id_, noun, description, id_example, query in loader.iter_anchors()
        }

        links: dict[str, Link] = {}
        for (
            id_,
            id_from,
            id_to,
            sentence,
            description,
            query,
            anchor1_link_column_name,
            anchor2_link_column_name,
            has_direction,
        ) in loader.iter_links():
            anchor_from = anchors.get(id_from)
            anchor_to = anchors.get(id_to)
            if anchor_from is None or anchor_to is None:
                logger.error(f'Link {sentence} has invalid connection "{anchor_from} - {anchor_to}"')
                continue

            links[id_] = Link(
                anchor_from=anchor_from,
                anchor_to=anchor_to,
                anchor_from_link_attr_name=anchor1_link_column_name,
                anchor_to_link_attr_name=anchor2_link_column_name,
                sentence=sentence,
                description=description,
                query=query,
                has_direction=has_direction,
                attributes=[],
            )

        attrs: list[Attribute] = []
        for (
            id_,
            a_id,
            l_id,
            name,
            description,
            example,
            query,
            dtype,
            embeddable,
            embed_threshold,
        ) in loader.iter_attrs():
            attr = Attribute(
                name=name,
                description=description,
                example=example,
                query=query,
                dtype=dtype,
                embeddable=embeddable,
                embed_threshold=embed_threshold,
                meta={"db_id": id_},
            )
            attrs.append(attr)
            if a_id:
                anchor = anchors.get(a_id)
                if anchor:
                    anchor.attributes.append(attr)
                    if not attr.name.startswith(anchor.noun):
                        logger.warning(f"Violation of naming convention: anchor {anchor.noun} has attr {attr.name}")
                else:
                    logger.error(f"Anchor {a_id} for attr {attr.name} not found")

            if l_id:
                link = links.get(l_id)
                if link:
                    link.attributes.append(attr)
                else:
                    logger.error(f"Link {l_id} for attr {attr.name} not found")

        queries = [Query(name, example) for name, example in loader.iter_queries()]

        conversation_lifecycle = [
            ConversationLifecycleEvent(event, text) for event, text in loader.iter_conversation_lifecycle_events()
        ]

        prompts = [Prompt(name, text) for name, text in loader.iter_prompts()]

        return cls(
            anchors=list(anchors.values()),
            links=list(links.values()),
            attrs=attrs,
            queries=queries,
            conversation_lifecycle=conversation_lifecycle,
            prompts=prompts,
        )

    def embeddable_attributes(self) -> dict[str, dict]:
        return {
            attr.name: {"noun": anchor.noun, "th": attr.embed_threshold}
            for anchor in self.anchors
            for attr in anchor.attributes
            if attr.embeddable
        } | {
            attr.name: {"link": link.sentence, "th": attr.embed_threshold}
            for link in self.links
            for attr in link.attributes
            if attr.embeddable
        }

    def conversation_lifecycle_events(self) -> dict[str, str]:
        return {cl.event: cl.text for cl in self.conversation_lifecycle}

    def prompt_templates(self) -> dict[str, str]:
        return {p.name: p.text for p in self.prompts}

    def vector_indices(self) -> list[tuple[str, str]]:
        a_i = [(anchor.noun, attr.name) for anchor in self.anchors for attr in anchor.attributes if attr.embeddable]
        l_i = [(link.sentence, attr.name) for link in self.links for attr in link.attributes if attr.embeddable]
        # todo links as well
        return a_i + l_i

    def anchor_links(self, anchor_noun: str) -> list[Link]:
        return [
            link
            for link in self.links
            if (link.anchor_from.noun == anchor_noun and link.anchor_from_link_attr_name)
            or (link.anchor_to.noun == anchor_noun and link.anchor_to_link_attr_name)
        ]

    def to_text_descr(self) -> str:
        dm_templates = self.prompt_templates()

        anchor_descr = "\n".join(
            dm_templates.get("dm_anchor_descr_template", dm_anchor_descr_template).format(anchor=anchor)
            for anchor in self.anchors
        )

        anchor_attrs_descr = "\n".join(
            dm_templates.get("dm_attr_descr_template", dm_attr_descr_template).format(anchor=anchor, attr=attr)
            for anchor in self.anchors
            for attr in anchor.attributes
        )

        link_descr = "\n".join(
            dm_templates.get("dm_link_descr_template", dm_link_descr_template).format(link=link) for link in self.links
        )

        link_attrs_descr = "\n".join(
            dm_templates.get("dm_link_attr_descr_template", dm_link_attr_descr_template).format(link=link, attr=attr)
            for link in self.links
            for attr in link.attributes
        )

        queries_descr = "\n".join(
            dm_templates.get("dm_query_descr_template", dm_query_descr_template).format(query=query)
            for query in self.queries
        )

        dm_template = dm_templates.get("dm_descr_template", dm_descr_template)

        return dm_template.format(
            anchors=anchor_descr,
            anchor_attrs=anchor_attrs_descr,
            links=link_descr,
            link_attrs=link_attrs_descr,
            queries=queries_descr,
        )


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
