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

        with self._db_engine.connect() as conn:
            result = conn.execute(select(anchors_table))
            anchors = []
            for row in result:
                anchors.append(
                    Anchor(
                        noun=row.noun,
                        description=row.description,
                        id_example=row.id_example,
                        query=row.query,
                        attributes=[],  # Attributes are loaded separately
                    )
                )
            return anchors

    def _get_links(self, anchors_dict: dict[str, Anchor] | None = None) -> list[Link]:
        """Read links from dm_links table."""
        links_table = sa.Table(
            "dm_links",
            sa.MetaData(),
            autoload_with=self._db_engine,
        )

        if anchors_dict is None:
            anchors_dict = {anchor.noun: anchor for anchor in self._get_anchors()}

        with self._db_engine.connect() as conn:
            result = conn.execute(select(links_table))
            links = []
            for row in result:
                anchor_from = anchors_dict.get(row.anchor1)
                anchor_to = anchors_dict.get(row.anchor2)
                if anchor_from is None or anchor_to is None:
                    logger.error(f'Link {row.sentence} has invalid connection "{row.anchor1} - {row.anchor2}"')
                    continue

                links.append(
                    Link(
                        anchor_from=anchor_from,
                        anchor_to=anchor_to,
                        anchor_from_link_attr_name=row.anchor1_link_column_name,
                        anchor_to_link_attr_name=row.anchor2_link_column_name,
                        sentence=row.sentence,
                        description=row.description,
                        query=row.query,
                        has_direction=bool(row.has_direction) if row.has_direction is not None else False,
                        attributes=[],  # Attributes are loaded separately
                    )
                )
            return links

    def _get_attributes(
        self, anchors_dict: dict[str, Anchor] | None = None, links_dict: dict[str, Link] | None = None
    ) -> list[Attribute]:
        """Read attributes from dm_attributes_v2 or dm_attributes table."""
        # Try dm_attributes_v2 first, fallback to dm_attributes
        try:
            attrs_table = sa.Table(
                "dm_attributes_v2",
                sa.MetaData(),
                autoload_with=self._db_engine,
            )
        except sa_exc.NoSuchTableError:
            attrs_table = sa.Table(
                "dm_attributes",
                sa.MetaData(),
                autoload_with=self._db_engine,
            )

        if anchors_dict is None:
            anchors_dict = {anchor.noun: anchor for anchor in self._get_anchors()}
        if links_dict is None:
            links_dict = {}
            for link in self._get_links(anchors_dict):
                # Use sentence as key for links
                links_dict[link.sentence] = link

        with self._db_engine.connect() as conn:
            result = conn.execute(select(attrs_table))
            attrs = []
            for row in result:
                attr = Attribute(
                    name=row.attribute_name,
                    description=row.description,
                    example=row.data_example,
                    query=row.query,
                    dtype=row.dtype,
                    embeddable=bool(row.embeddable) if row.embeddable is not None else False,
                    embed_threshold=float(row.embed_threshold) if row.embed_threshold is not None else 0.0,
                    meta={"db_id": row.attribute_name},
                )
                attrs.append(attr)

                # Associate with anchor if specified
                if row.anchor:
                    anchor = anchors_dict.get(row.anchor)
                    if anchor:
                        anchor.attributes.append(attr)
                        if not attr.name.startswith(anchor.noun):
                            logger.warning(f"Violation of naming convention: anchor {anchor.noun} has attr {attr.name}")
                    else:
                        logger.error(f"Anchor {row.anchor} for attr {attr.name} not found")

                # Associate with link if specified
                if row.link:
                    link_obj: Link | None = links_dict.get(row.link)
                    if link_obj is not None:
                        link_obj.attributes.append(attr)
                    else:
                        logger.error(f"Link {row.link} for attr {attr.name} not found")

            return attrs

    @property
    def anchors(self) -> list[Anchor]:
        return self._get_anchors()

    @property
    def links(self) -> list[Link]:
        return self._get_links()

    @property
    def attrs(self) -> list[Attribute]:
        return self._get_attributes()

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
                return [Query(name=row.name, example=row.example) for row in result]
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

    def embeddable_attributes(self) -> dict[str, dict]:
        # todo anchors / edges
        anchors = self.anchors
        links = self.links

        result = {
            attr.name: {"noun": anchor.noun, "th": attr.embed_threshold}
            for anchor in anchors
            for attr in anchor.attributes
            if attr.embeddable
        }
        result.update(
            {
                attr.name: {"link": link.sentence, "th": attr.embed_threshold}
                for link in links
                for attr in link.attributes
                if attr.embeddable
            }
        )
        return result

    def conversation_lifecycle_events(self) -> dict[str, str]:
        return {cl.event: cl.text for cl in self.conversation_lifecycle}

    def prompt_templates(self) -> dict[str, str]:
        return {p.name: p.text for p in self.prompts}

    def vector_indices(self) -> list[tuple[str, str]]:
        anchors = self.anchors
        links = self.links
        # todo links as well
        a_i = [(anchor.noun, attr.name) for anchor in anchors for attr in anchor.attributes if attr.embeddable]
        l_i = [(link.sentence, attr.name) for link in links for attr in link.attributes if attr.embeddable]
        return a_i + l_i

    def anchor_links(self, anchor_noun: str) -> list[Link]:
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
