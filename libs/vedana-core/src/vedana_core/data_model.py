import logging
from dataclasses import dataclass

import sqlalchemy.ext.asyncio as sa_aio
from sqlalchemy import select
from vedana_etl.catalog import (
    dm_anchor_attributes,
    dm_anchors,
    dm_conversation_lifecycle,
    dm_link_attributes,
    dm_links,
    dm_prompts,
    dm_queries,
)

logger = logging.getLogger(__name__)


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
    DataModel, read from SQL tables at runtime
    """

    def __init__(self, sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession]) -> None:
        self.sessionmaker = sessionmaker

    @classmethod
    def create(cls, sessionmaker) -> "DataModel":
        return cls(sessionmaker=sessionmaker)

    async def get_anchors(self) -> list[Anchor]:
        """Read anchors from dm_anchors table."""
        # .data_table is TableStoreDB's attribute
        anchors_table = dm_anchors.store.data_table  # type: ignore[attr-defined]
        anchors_attr_table = dm_anchor_attributes.store.data_table  # type: ignore[attr-defined]

        async with self.sessionmaker() as session:
            join_query = select(
                anchors_table.c.noun,
                anchors_table.c.description.label("anchor_description"),
                anchors_table.c.id_example,
                anchors_table.c.query.label("anchor_query"),
                anchors_attr_table.c.attribute_name,
                anchors_attr_table.c.description.label("attr_description"),
                anchors_attr_table.c.data_example,
                anchors_attr_table.c.embeddable,
                anchors_attr_table.c.query.label("attr_query"),
                anchors_attr_table.c.dtype,
                anchors_attr_table.c.embed_threshold,
            ).select_from(
                anchors_table.join(  # left join
                    anchors_attr_table,
                    anchors_table.c.noun == anchors_attr_table.c.anchor,
                    isouter=True,
                )
            )
            result = (await session.execute(join_query)).fetchall()

            anchors = {}
            for row in result:
                noun = row.noun
                if noun not in anchors:
                    anchors[noun] = Anchor(
                        noun=noun,
                        description=row.anchor_description,
                        id_example=row.id_example,
                        query=row.anchor_query,
                        attributes=[],
                    )

                # Add attribute if it exists (attribute_name will be None for anchors without attributes)
                if row.attribute_name is not None:
                    anchors[noun].attributes.append(
                        Attribute(
                            name=row.attribute_name,
                            description=row.attr_description if row.attr_description else "",
                            example=row.data_example if row.data_example else "",
                            embeddable=row.embeddable if row.embeddable is not None else False,
                            query=row.attr_query if row.attr_query else "",
                            dtype=row.dtype if row.dtype else "",
                            embed_threshold=row.embed_threshold if row.embed_threshold is not None else 1.0,
                        )
                    )

            return list(anchors.values())

    async def get_links(self, anchors_dict: dict[str, Anchor] | None = None) -> list[Link]:
        """Read links from dm_links table."""
        links_table = dm_links.store.data_table  # type: ignore[attr-defined]
        links_attr_table = dm_link_attributes.store.data_table  # type: ignore[attr-defined]

        if anchors_dict is None:
            anchors = await self.get_anchors()
            anchors_dict = {anchor.noun: anchor for anchor in anchors}

        async with self.sessionmaker() as session:
            join_query = select(
                links_table.c.anchor1,
                links_table.c.anchor2,
                links_table.c.sentence,
                links_table.c.description.label("link_description"),
                links_table.c.query.label("link_query"),
                links_table.c.anchor1_link_column_name,
                links_table.c.anchor2_link_column_name,
                links_table.c.has_direction,
                links_attr_table.c.attribute_name,
                links_attr_table.c.description.label("attr_description"),
                links_attr_table.c.data_example,
                links_attr_table.c.embeddable,
                links_attr_table.c.query.label("attr_query"),
                links_attr_table.c.dtype,
                links_attr_table.c.embed_threshold,
            ).select_from(
                links_table.join(  # left join
                    links_attr_table,
                    links_table.c.sentence == links_attr_table.c.link,
                    isouter=True,
                )
            )

            result = (await session.execute(join_query)).fetchall()

            links = {}
            for row in result:
                sentence = row.sentence
                if sentence not in links:
                    anchor_from = anchors_dict.get(row.anchor1)
                    anchor_to = anchors_dict.get(row.anchor2)
                    if anchor_from is None or anchor_to is None:
                        logger.error(f'Link {sentence} has invalid connection "{row.anchor1} - {row.anchor2}"')
                        continue

                    links[sentence] = Link(
                        anchor_from=anchor_from,
                        anchor_to=anchor_to,
                        anchor_from_link_attr_name=row.anchor1_link_column_name,
                        anchor_to_link_attr_name=row.anchor2_link_column_name,
                        sentence=sentence,
                        description=row.link_description,
                        query=row.link_query,
                        has_direction=bool(row.has_direction) if row.has_direction is not None else False,
                        attributes=[],
                    )

                # Add attribute if it exists (attribute_name will be None for anchors without attributes)
                if row.attribute_name is not None:
                    links[sentence].attributes.append(
                        Attribute(
                            name=row.attribute_name,
                            description=row.attr_description if row.attr_description else "",
                            example=row.data_example if row.data_example else "",
                            embeddable=row.embeddable if row.embeddable is not None else False,
                            query=row.attr_query if row.attr_query else "",
                            dtype=row.dtype if row.dtype else "",
                            embed_threshold=row.embed_threshold if row.embed_threshold is not None else 1.0,
                        )
                    )

            return list(links.values())

    async def get_queries(self) -> list[Query]:
        try:
            queries_table = dm_queries.store.data_table  # type: ignore[attr-defined]
            async with self.sessionmaker() as session:
                result = (await session.execute(select(queries_table))).fetchall()
                return [Query(name=row.query_name, example=row.query_example) for row in result]
        except Exception:
            return []

    async def get_conversation_lifecycle_events(self) -> list[ConversationLifecycleEvent]:
        try:
            lifecycle_table = dm_conversation_lifecycle.store.data_table  # type: ignore[attr-defined]
            async with self.sessionmaker() as session:
                result = (await session.execute(select(lifecycle_table))).fetchall()
                return [ConversationLifecycleEvent(event=row.event, text=row.text) for row in result]
        except Exception:
            return []

    async def conversation_lifecycle_events(self) -> dict[str, str]:
        cl = await self.get_conversation_lifecycle_events()
        return {c.event: c.text for c in cl}

    async def get_prompts(self) -> list[Prompt]:
        try:
            prompts_table = dm_prompts.store.data_table  # type: ignore[attr-defined]
            async with self.sessionmaker() as session:
                result = (await session.execute(select(prompts_table))).fetchall()
                return [Prompt(name=row.name, text=row.text) for row in result]
        except Exception:
            return []

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
        attribute_names: list[str] | None = None,
        query_names: list[str] | None = None,
    ) -> str:
        """Create a text description of the data model, optionally filtered.

        Args:
            anchor_nouns: List of anchor nouns to include. If None, includes all.
            link_sentences: List of link sentences to include. If None, includes all.
            attribute_names: List of attribute names to include. If None, includes all.
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
        attr_set = set(attribute_names) if attribute_names is not None else None
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
            if attr_set is None or attr.name in attr_set
        )

        link_descr = "\n".join(
            dm_templates.get("dm_link_descr_template", dm_link_descr_template).format(link=link)
            for link in filtered_links
        )

        link_attrs_descr = "\n".join(
            dm_templates.get("dm_link_attr_descr_template", dm_link_attr_descr_template).format(link=link, attr=attr)
            for link in filtered_links
            for attr in link.attributes
            if attr_set is None or attr.name in attr_set
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
