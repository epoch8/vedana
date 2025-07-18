import abc
import json
import logging
import sqlite3
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import grist_api
import pandas as pd
import requests
from graph import Graph

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


class DataModelLoader(abc.ABC):
    @abc.abstractmethod
    def iter_anchors(self) -> Iterable[tuple]: ...

    @abc.abstractmethod
    def iter_links(self) -> Iterable[tuple]: ...

    @abc.abstractmethod
    def iter_attrs(self) -> Iterable[tuple]: ...

    @abc.abstractmethod
    def iter_queries(self) -> Iterable[tuple]: ...

    @abc.abstractmethod
    def iter_conversation_lifecycle_events(self) -> Iterable[tuple]: ...

    def close(self) -> None: ...


class DataModel:
    def __init__(
        self,
        anchors: list[Anchor],
        links: list[Link],
        attrs: list[Attribute],
        queries: list[Query],
        conversation_lifecycle: list[ConversationLifecycleEvent],
    ) -> None:
        self.anchors = anchors
        self.links = links
        self.attrs = attrs
        self.queries = queries
        self.conversation_lifecycle = conversation_lifecycle

    @classmethod
    def load_sqlite(cls, db_path: Path) -> "DataModel":
        with closing(SqliteDataModelLoader(db_path)) as loader:
            return cls._load(loader)

    @classmethod
    def load_grist_online(cls, doc_id: str, grist_server: str, api_key: str | None = None) -> "DataModel":
        with closing(GristOnlineDataModelLoader(doc_id=doc_id, grist_server=grist_server, api_key=api_key)) as loader:
            return cls._load(loader)

    @classmethod
    def _load(cls, loader: DataModelLoader) -> "DataModel":
        # TODO id_example?
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

        return cls(
            anchors=list(anchors.values()),
            links=list(links.values()),
            attrs=attrs,
            queries=queries,
            conversation_lifecycle=conversation_lifecycle,
        )

    def embeddable_attributes(self) -> dict[str, dict]:
        return {
            attr.name: {"noun": anchor.noun, "th": attr.embed_threshold}
            for anchor in self.anchors
            for attr in anchor.attributes
            if attr.embeddable and attr.dtype == "str"
        }

    def conversation_lifecycle_events(self) -> dict[str, str]:
        return {cl.event: cl.text for cl in self.conversation_lifecycle}

    def vector_indices(self) -> dict[str, str]:
        return {
            anchor.noun: attr.name
            for anchor in self.anchors
            for attr in anchor.attributes
            if attr.embeddable and attr.dtype == "str"
        }

    def anchor_links(self, anchor_noun: str) -> list[Link]:
        return [
            l
            for l in self.links
            if (l.anchor_from.noun == anchor_noun and l.anchor_from_link_attr_name)
            or (l.anchor_to.noun == anchor_noun and l.anchor_to_link_attr_name)
        ]

    def to_text_descr(self) -> str:
        anchor_descr = "\n".join(
            f"- {anchor.noun}: {anchor.description}; пример ID: {anchor.id_example};"
            f" запрос для получения: {anchor.query}"
            for anchor in self.anchors
        )

        anchor_attrs_descr = "\n".join(
            f"- {anchor.noun}.{attr.name}: {attr.description}; пример: {attr.example};"
            f" запрос для получения: {attr.query}"
            for anchor in self.anchors
            for attr in anchor.attributes
        )

        link_descr = "\n".join(
            f"- {link.sentence}: {link.description}; пример запроса: {link.query}" for link in self.links
        )

        link_attrs_descr = "\n".join(
            f"- {link.sentence}.{attr.name}: {attr.description}; пример: {attr.example};"
            f" запрос для получения: {attr.query}"
            for link in self.links
            for attr in link.attributes
        )

        queries_descr = "\n".join(f"- {query.name}\n{query.example}" for query in self.queries)

        return text_description_template.format(
            anchors=anchor_descr,
            anchor_attrs=anchor_attrs_descr,
            links=link_descr,
            link_attrs=link_attrs_descr,
            queries=queries_descr,
        )

    def to_dict(self) -> dict:
        """serialize DataModel"""

        anchors = [
            {
                "noun": a.noun,
                "description": a.description,
                "id_example": a.id_example,
                "query": a.query,
            }
            for a in self.anchors
        ]

        links = [
            {
                "anchor_from": l.anchor_from.noun,
                "anchor_to": l.anchor_to.noun,
                "sentence": l.sentence,
                "description": l.description,
                "query": l.query,
                "has_direction": l.has_direction,
                "anchor_from_link_attr_name": l.anchor_from_link_attr_name,
                "anchor_to_link_attr_name": l.anchor_to_link_attr_name,
            }
            for l in self.links
        ]

        # Flatten attributes and keep mapping to reconstruct later
        attrs: list[dict] = []
        for a in self.anchors:
            for attr in a.attributes:
                attrs.append(
                    {
                        "name": attr.name,
                        "description": attr.description,
                        "example": attr.example,
                        "query": attr.query,
                        "dtype": attr.dtype,
                        "embeddable": attr.embeddable,
                        "embed_threshold": attr.embed_threshold,
                        "anchor_noun": a.noun,
                        "link_sentence": None,
                    }
                )

        for l in self.links:
            for attr in l.attributes:
                attrs.append(
                    {
                        "name": attr.name,
                        "description": attr.description,
                        "example": attr.example,
                        "query": attr.query,
                        "dtype": attr.dtype,
                        "embeddable": attr.embeddable,
                        "embed_threshold": attr.embed_threshold,
                        "anchor_noun": None,
                        "link_sentence": l.sentence,
                    }
                )

        queries = [{"name": q.name, "example": q.example} for q in self.queries]

        return {
            "anchors": anchors,
            "links": links,
            "attrs": attrs,
            "queries": queries,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict) -> "DataModel":
        """Reconstruct DataModel"""

        # 1. Anchors
        anchors_map: dict[str, Anchor] = {}
        for a in d.get("anchors", []):
            anchor = Anchor(
                noun=a["noun"],
                description=a.get("description", ""),
                id_example=a.get("id_example", ""),
                query=a.get("query", ""),
                attributes=[],
            )
            anchors_map[anchor.noun] = anchor

        # 2. Links
        links_map: dict[str, Link] = {}
        for l in d.get("links", []):
            af_noun = l["anchor_from"]
            at_noun = l["anchor_to"]
            link_obj = Link(
                anchor_from=anchors_map[af_noun],
                anchor_to=anchors_map[at_noun],
                sentence=l["sentence"],
                description=l.get("description", ""),
                query=l.get("query", ""),
                attributes=[],
                has_direction=l.get("has_direction", False),
                anchor_from_link_attr_name=l.get("anchor_from_link_attr_name", ""),
                anchor_to_link_attr_name=l.get("anchor_to_link_attr_name", ""),
            )
            # Use sentence as unique key for quick lookup
            links_map[link_obj.sentence] = link_obj

        # 3. Attributes – create objects, attach to correct anchor / link
        attrs: list[Attribute] = []
        for a in d.get("attrs", []):
            attr_obj = Attribute(
                name=a["name"],
                description=a.get("description", ""),
                example=a.get("example", ""),
                query=a.get("query", ""),
                dtype=a.get("dtype", "str"),
                embeddable=a.get("embeddable", False),
                embed_threshold=a.get("embed_threshold", 0),
                meta={},
            )
            attrs.append(attr_obj)
            anchor_noun = a.get("anchor_noun")
            link_sentence = a.get("link_sentence")
            if anchor_noun and anchor_noun in anchors_map:
                anchors_map[anchor_noun].attributes.append(attr_obj)
            elif link_sentence and link_sentence in links_map:
                links_map[link_sentence].attributes.append(attr_obj)

        # 4. Queries
        queries = [Query(q["name"], q.get("example", "")) for q in d.get("queries", [])]

        # 5. ConversationLifecycle
        cl = [ConversationLifecycleEvent(c["event"], c.get("text", "")) for c in d.get("conversation_lifecycle", [])]

        return cls(
            anchors=list(anchors_map.values()),
            links=list(links_map.values()),
            attrs=attrs,
            queries=queries,
            conversation_lifecycle=cl,
        )

    @classmethod
    def from_json(cls, json_str: str) -> "DataModel":
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load_from_graph(cls, graph: "Graph") -> "DataModel | None":
        try:
            res = list(
                graph.execute_ro_cypher_query(
                    "MATCH (dm:DataModel {id: 'data_model'}) RETURN dm.content AS content LIMIT 1"
                )
            )
        except Exception:
            return None
        if not res:
            return None
        content = res[0].get("content")
        if not content:
            return None
        return cls.from_json(content)


text_description_template = """\
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


class SqliteDataModelLoader(DataModelLoader):
    def __init__(self, db_path: Path) -> None:
        super().__init__()
        self._conn = sqlite3.connect(db_path)

    def iter_anchors(self) -> Iterable[tuple]:
        yield from self._conn.execute("SELECT id, noun, description, id_example, query FROM anchors where noun != ''")

    def iter_links(self) -> Iterable[tuple]:
        yield from self._conn.execute(
            "SELECT id, anchor1, anchor2, sentence, description, query,"
            " anchor1_link_column_name, anchor2_link_column_name, has_direction"
            " FROM Links WHERE Sentence != ''"
        )

    def iter_attrs(self) -> Iterable[tuple]:
        yield from self._conn.execute(
            "SELECT id, Anchor, Link, Attribute_Name, description,"
            " Data_example, Query, Embeddable FROM Attributes2"
            " where Attribute_Name != ''"
        )

    def iter_queries(self) -> Iterable[tuple]:
        yield from self._conn.execute("SELECT query_name, query_example FROM Queries WHERE query_name != ''")

    def close(self) -> None:
        self._conn.close()


class GristOnlineDataModelLoader(DataModelLoader):
    def __init__(self, grist_server: str, doc_id: str, api_key: str | None = None) -> None:
        super().__init__()
        self._client = grist_api.GristDocAPI(doc_id, server=grist_server, api_key=api_key)

    def iter_anchors(self) -> Iterable[tuple]:
        anchors_df = self.get_table("Anchors")
        anchors_df = anchors_df[["id", "noun", "description", "id_example", "query"]]
        for row in anchors_df.itertuples(index=False):
            if row.noun:
                yield row

    def iter_links(self) -> Iterable[tuple]:
        anchors_df = self.get_table("Anchors")
        anchors_df = anchors_df.set_index("id")["noun"].squeeze()  # get id <--> noun mapping for resolving links
        df = self.get_table("Links")
        df = df[
            [
                "id",
                "anchor1",
                "anchor2",
                "sentence",
                "description",
                "query",
                "anchor1_link_column_name",
                "anchor2_link_column_name",
                "has_direction",
            ]
        ]
        df["anchor1"] = df["anchor1"].apply(lambda x: anchors_df.get(x, x))
        df["anchor2"] = df["anchor2"].apply(lambda x: anchors_df.get(x, x))
        for row in df.itertuples(index=False):
            if all([row.sentence, row.anchor1, row.anchor2]):
                yield row

    def iter_attrs(self) -> Iterable[tuple]:
        anchors_df = self.get_table("Anchors")
        anchors_df = anchors_df.set_index("id")["noun"].squeeze()  # get id <--> noun mapping for resolving links
        df = self.get_table("Attributes")
        df = df[
            [
                "id",
                "anchor",
                "link",
                "attribute_name",
                "description",
                "data_example",
                "query",
                "dtype",
                "embeddable",
                "embed_threshold",
            ]
        ]
        df["anchor"] = df["anchor"].apply(lambda x: anchors_df.get(x, x))
        for row in df.itertuples(index=False):
            if row.attribute_name:
                yield row

    def iter_queries(self) -> Iterable[tuple]:
        df = self.get_table("Queries")
        df = df[["query_name", "query_example"]]
        for row in df.itertuples(index=False):
            yield row

    def iter_conversation_lifecycle_events(self) -> Iterable[tuple]:
        try:
            df = self.get_table("ConversationLifecycle")
            df = df[["event", "text"]]
            for row in df.dropna().itertuples(index=False):
                yield row
        except requests.exceptions.HTTPError:
            logger.warning("ConversationLifecycle table not found in Grist document")

    def _list_table_columns(self, table_name: str) -> list[str]:
        resp = self._client.columns(table_name)
        if not resp:
            return []
        return [column["id"] for column in resp.json()["columns"]]

    def get_table(self, table_name: str) -> pd.DataFrame:
        columns = self._list_table_columns(table_name) + ["id"]
        rows = self._client.fetch_table(table_name)
        df = pd.DataFrame(data=rows)
        df = df[columns]
        df.columns = [col.lower() for col in df.columns]
        return df


class CsvDataModelLoader(DataModelLoader):
    def __init__(self, csv_dir: Path) -> None:
        super().__init__()
        self.csv_dir = Path(csv_dir)
        self._anchors = pd.read_csv(self.csv_dir / "anchors.csv")
        self._links = pd.read_csv(self.csv_dir / "links.csv")
        self._attrs = pd.read_csv(self.csv_dir / "attributes.csv")
        self._queries = pd.read_csv(self.csv_dir / "queries.csv")

    def iter_anchors(self) -> Iterable[tuple]:
        for row in self._anchors.itertuples(index=False):
            yield (
                getattr(row, "id", None),
                getattr(row, "noun", None),
                getattr(row, "description", None),
                getattr(row, "id_example", None),
                getattr(row, "query", None),
            )

    def iter_links(self) -> Iterable[tuple]:
        for row in self._links.itertuples(index=False):
            yield (
                getattr(row, "id", None),
                getattr(row, "anchor1", None),
                getattr(row, "anchor2", None),
                getattr(row, "sentence", None),
                getattr(row, "description", None),
                getattr(row, "query", None),
                getattr(row, "anchor1_link_column_name", None),
                getattr(row, "anchor2_link_column_name", None),
                getattr(row, "has_direction", None),
            )

    def iter_attrs(self) -> Iterable[tuple]:
        for row in self._attrs.itertuples(index=False):
            yield (
                getattr(row, "id", None),
                getattr(row, "anchor", None),
                getattr(row, "link", None),
                getattr(row, "attribute_name", None),
                getattr(row, "description", None),
                getattr(row, "data_example", None),
                getattr(row, "query", None),
                getattr(row, "embeddable", None),
            )

    def iter_queries(self) -> Iterable[tuple]:
        for row in self._queries.itertuples(index=False):
            yield (
                getattr(row, "query_name", None),
                getattr(row, "query_example", None),
            )

    def close(self) -> None:
        pass


def main():
    from settings import settings as S

    data_model2 = DataModel.load_grist_online(
        S.grist_data_model_doc_id, api_key=S.grist_api_key, grist_server=S.grist_server_url
    )
    print("DataModel from grist:")
    print(data_model2.to_text_descr())


if __name__ == "__main__":
    sys.exit(main())
