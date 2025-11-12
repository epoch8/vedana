import logging
from dataclasses import dataclass
from typing import Any

import sqlalchemy as sa
from sqlalchemy import exc as sa_exc
from sqlalchemy import select

from vedana_core.db import get_db_engine

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

    def __init__(self, db_engine: sa.Engine | None = None) -> None:
        self._db_engine = db_engine or get_db_engine()

    @classmethod
    def create(cls, db_engine: sa.Engine | None = None) -> "DataModel":
        return cls(db_engine=db_engine)

    def _get_anchors(self) -> list[Anchor]:
        """Read anchors from dm_anchors table."""
        anchors_table = sa.Table(
            "dm_anchors",
            sa.MetaData(),
            autoload_with=self._db_engine,
        )
        anchors_attr_table = sa.Table(
            "dm_anchor_attributes",
            sa.MetaData(),
            autoload_with=self._db_engine,
        )

        with self._db_engine.connect() as conn:
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
            result = conn.execute(join_query)

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

    def _get_links(self, anchors_dict: dict[str, Anchor] | None = None) -> list[Link]:
        """Read links from dm_links table."""
        links_table = sa.Table(
            "dm_links",
            sa.MetaData(),
            autoload_with=self._db_engine,
        )
        links_attr_table = sa.Table(
            "dm_link_attributes",
            sa.MetaData(),
            autoload_with=self._db_engine,
        )

        if anchors_dict is None:
            anchors_dict = {anchor.noun: anchor for anchor in self._get_anchors()}

        with self._db_engine.connect() as conn:
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
            result = conn.execute(join_query)

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

    @property
    def anchors(self) -> list[Anchor]:
        return self._get_anchors()

    @property
    def links(self) -> list[Link]:
        return self._get_links()

    @property
    def queries(self) -> list[Query]:
        try:
            queries_table = sa.Table(
                "dm_queries",
                sa.MetaData(),
                autoload_with=self._db_engine,
            )
            with self._db_engine.connect() as conn:
                result = conn.execute(select(queries_table))
                return [Query(name=row.query_name, example=row.query_example) for row in result]
        except sa_exc.NoSuchTableError:
            return []

    @property
    def conversation_lifecycle(self) -> list[ConversationLifecycleEvent]:
        try:
            lifecycle_table = sa.Table(
                "dm_conversation_lifecycle",
                sa.MetaData(),
                autoload_with=self._db_engine,
            )
            with self._db_engine.connect() as conn:
                result = conn.execute(select(lifecycle_table))
                return [ConversationLifecycleEvent(event=row.event, text=row.text) for row in result]
        except sa_exc.NoSuchTableError:
            return []

    @property
    def prompts(self) -> list[Prompt]:
        try:
            prompts_table = sa.Table(
                "dm_prompts",
                sa.MetaData(),
                autoload_with=self._db_engine,
            )
            with self._db_engine.connect() as conn:
                result = conn.execute(select(prompts_table))
                return [Prompt(name=row.name, text=row.text) for row in result]
        except sa_exc.NoSuchTableError:
            return []

    def conversation_lifecycle_events(self) -> dict[str, str]:
        return {cl.event: cl.text for cl in self.conversation_lifecycle}

    def prompt_templates(self) -> dict[str, str]:
        return {p.name: p.text for p in self.prompts}

    def vector_indices(self) -> list[tuple[str, str, str, float]]:
        """
        returns list
        ("anchor", anchor.noun, anchor.attribute, anchor.th) +
        ("edge", link.sentence, link.attribute, link.th)
        for all embeddable attributes
        """
        anchors = self.anchors
        links = self.links
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

    def anchor_links(self, anchor_noun: str) -> list[Link]:
        """all links that connect to/from this anchor"""
        links = self.links
        return [
            link
            for link in links
            if (link.anchor_from.noun == anchor_noun and link.anchor_from_link_attr_name)
            or (link.anchor_to.noun == anchor_noun and link.anchor_to_link_attr_name)
        ]

    def to_text_descr(self) -> str:
        anchors = self.anchors
        links = self.links
        queries = self.queries
        dm_templates = self.prompt_templates()

        anchor_descr = "\n".join(
            dm_templates.get("dm_anchor_descr_template", dm_anchor_descr_template).format(anchor=anchor)
            for anchor in anchors
        )

        anchor_attrs_descr = "\n".join(
            dm_templates.get("dm_attr_descr_template", dm_attr_descr_template).format(anchor=anchor, attr=attr)
            for anchor in anchors
            for attr in anchor.attributes
        )

        link_descr = "\n".join(
            dm_templates.get("dm_link_descr_template", dm_link_descr_template).format(link=link) for link in links
        )

        link_attrs_descr = "\n".join(
            dm_templates.get("dm_link_attr_descr_template", dm_link_attr_descr_template).format(link=link, attr=attr)
            for link in links
            for attr in link.attributes
        )

        queries_descr = "\n".join(
            dm_templates.get("dm_query_descr_template", dm_query_descr_template).format(query=query)
            for query in queries
        )

        dm_template = dm_templates.get("dm_descr_template", dm_descr_template)

        return dm_template.format(
            anchors=anchor_descr,
            anchor_attrs=anchor_attrs_descr,
            links=link_descr,
            link_attrs=link_attrs_descr,
            queries=queries_descr,
        )


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
