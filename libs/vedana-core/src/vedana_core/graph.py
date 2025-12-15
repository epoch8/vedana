import abc
import json
import logging
import re
import sys
from typing import Any, Dict, Iterable, Set, cast

import aioitertools as aioit
import neo4j
import numpy as np
import typing_extensions as te
from neo4j import AsyncGraphDatabase, EagerResult, RoutingControl
from opentelemetry import trace
from pgvector.asyncpg import register_vector
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from vedana_core.db import get_sessionmaker

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

Record = neo4j.Record


class Graph(abc.ABC):
    async def add_node(
        self,
        node_id: str,
        labels: Set[str],
        properties: dict[str, Any] | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> None:
        raise NotImplementedError

    async def add_edge(self, from_id: str, to_id: str, type_: str, attrs: Dict[str, Any] | None) -> None:
        raise NotImplementedError

    async def number_of_nodes(self) -> int:
        raise NotImplementedError

    async def number_of_edges(self) -> int:
        raise NotImplementedError

    async def run_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterable[Record]:
        raise NotImplementedError

    async def get_existing_node_types(self) -> Iterable[list[str]]:
        raise NotImplementedError

    async def create_full_text_search_index(self, label: str) -> None:
        raise NotImplementedError

    async def create_vector_search_index(self, label: str, prop_name: str, dimension: int) -> None:
        raise NotImplementedError

    async def create_snapshot(self) -> None:
        raise NotImplementedError

    async def llm_schema(self) -> str:
        raise NotImplementedError

    async def text_search(self, label: str, query: str, limit: int = 10) -> Iterable[Record]:
        raise NotImplementedError

    async def vector_search(
        self,
        label: str,
        prop_type: str,
        prop_name: str,
        embedding: np.ndarray | list[float],
        threshold: float,
        top_n: int = 10,
    ) -> list[Record]:
        raise NotImplementedError

    async def setup(self, *_, create_basic_indices: bool = True, **kwargs) -> None:
        # Set false to speedup import
        if create_basic_indices:
            await self.create_basic_indices()

    async def create_basic_indices(self) -> None: ...

    async def execute_ro_cypher_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterable[Record]:
        return await self.run_cypher(query, parameters, limit=limit)

    async def clear(self) -> None: ...

    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CypherGraph(Graph):
    async def add_node(
        self,
        node_id: str,
        labels: Set[str],
        properties: dict[str, Any] | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> None:
        query, params = self._add_node_cypher(node_id, labels, properties or {})
        await self.run_cypher(query, params)

    def _add_node_cypher(
        self,
        node_id: str,
        labels: Set[str],
        properties: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        labels_expr = escape_labels(labels)
        props = {
            **properties,
            "id": node_id,
        }
        # TODO escape
        pros_expr = ", ".join(f"{k}: ${k}" for k in props.keys())
        return (
            f"MERGE (n:{labels_expr} {{id: $id}}) SET n = {{{pros_expr}}} RETURN n",
            props,
        )

    async def add_edge(self, from_id: str, to_id: str, type_: str, attrs: Dict[str, Any] | None) -> None:
        query, params = self._add_edge_cypher(from_id, to_id, type_, attrs)
        await self.run_cypher(query, params)

    def _add_edge_cypher(
        self, from_id: str, to_id: str, type_: str, attrs: Dict[str, Any] | None
    ) -> tuple[str, dict[str, Any]]:
        attrs = attrs or {}
        labels_expr = escape_labels({type_})
        # attrs = {escape_cypher(k): v for k, v in attrs.items()}
        attrs_expr = ", ".join(f"{k}: ${k}" for k in attrs.keys() if k)
        params = {
            **attrs,
            "from_id": from_id,
            "to_id": to_id,
        }
        return (
            "MATCH (nf {id: $from_id}), (nt {id: $to_id}) "
            f"CREATE (nf)-[r:{labels_expr} {{{attrs_expr}}}]->(nt) RETURN r",
            params,
        )

    async def add_edges(self, edges: Iterable[tuple[str, str, dict]], **common_attrs) -> None:
        for edge_tuple in edges:
            from_id, to_id, attrs = edge_tuple
            attrs = {**common_attrs, **attrs}
            labels: Iterable[str] = attrs.pop("__labels__", [])
            type_ = next(iter(labels), "no_type")
            await self.add_edge(from_id, to_id, type_, attrs)

    async def number_of_nodes(self) -> int:
        res = await self.execute_ro_cypher_query("MATCH (n) RETURN count(*) as cnt")
        return next(iter(res))["cnt"]

    async def number_of_edges(self) -> int:
        res = await self.execute_ro_cypher_query("MATCH (f)-[]->(t) RETURN count(*) as cnt")
        return next(iter(res))["cnt"]

    async def get_existing_node_types(self) -> Iterable[list[str]]:
        res = await self.execute_ro_cypher_query("MATCH (n) RETURN DISTINCT labels(n) as l;")
        return [r["l"] for r in res]


# class NXGraph(Graph):
#     def __init__(self, graph: nx.Graph) -> None:
#         self.graph: nx.Graph = graph
#         self.gcypher = GrandCypher(self.graph)

#     def execute_ro_cypher_query(self, query: str) -> Iterable[Any]:
#         return self.gcypher.run(query)

#     def add_node(self, node_id: str, labels: Set[str], **attributes) -> None:
#         self.graph.add_node(node_id, __labels__=labels, **attributes)

#     def number_of_edges(self) -> int:
#         return self.graph.number_of_edges()

#     def clear(self) -> None:
#         self.graph.clear()


class MemgraphGraph(CypherGraph):
    def __init__(self, uri: str, user: str, pwd: str, db_name: str = "") -> None:
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd), database=db_name)
        # await self.driver.verify_connectivity()
        self.driver_uri = uri
        self.auth = (user, pwd)

    async def execute_ro_cypher_query(
        self, query: str, parameters: dict[str, Any] | None = None, limit: int | None = None
    ) -> Iterable[Record]:
        with tracer.start_as_current_span("memgraph.execute_ro_cypher_query") as span:
            span.set_attribute("memgraph.query", query)
            if parameters:
                span.set_attribute("memgraph.parameters", json.dumps(parameters))
            result: EagerResult = await self.driver.execute_query(query, parameters, routing_=RoutingControl.READ)

        return result.records

    async def run_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterable[Record]:
        with tracer.start_as_current_span("memgraph.run_cypher") as span:
            span.set_attribute("memgraph.query", query)
            if parameters:
                span.set_attribute("memgraph.parameters", json.dumps(parameters))
            if limit is not None:
                span.set_attribute("memgraph.limit", limit)

            async with self.driver.session() as session:
                result = await aioit.list(aioit.islice(await session.run(query, parameters), limit))

        return result

    async def add_node(
        self,
        node_id: str,
        labels: Set[str],
        properties: Dict[str, Any] | None = None,
        embeddings: Dict[str, np.ndarray] | None = None,
    ) -> None:
        if properties and embeddings:
            embed_props = {f"{prop_name}_embedding": v for prop_name, v in embeddings.items()}
            properties = {
                **properties,
                **embed_props,
            }
        await super().add_node(node_id, labels, properties, embeddings)

    async def llm_schema(self) -> str:
        res = await self.driver.execute_query("CALL llm_util.schema() YIELD schema RETURN schema")
        return res.records[0]["schema"]

    async def create_basic_indices(self, node_types=None) -> None:
        if not node_types:
            node_types = await self.get_existing_node_types()
        for label in node_types:
            await self.create_node_prop_index(set(label), "id", unique=True)

    async def clear(self) -> None:
        async with self.driver.session() as session:
            res = await session.run("CALL vector_search.show_index_info() YIELD index_name RETURN *")

            async for (idx_name,) in res:
                await session.run(f"DROP VECTOR INDEX {escape_cypher(idx_name)}")
            idx_name_re = re.compile(r"\(name:\s(.+?)\)")
            async for row in await session.run(cast(te.LiteralString, "SHOW INDEX INFO")):
                index_type = row["index type"]
                idx_name = next(iter(idx_name_re.findall(index_type)), None)
                if not idx_name:
                    continue
                await session.run(f"DROP TEXT INDEX {escape_cypher(idx_name)}")
            await session.run("CALL schema.assert({}, {}, {}, true) YIELD action, key, keys, label, unique")
            await session.run("MATCH (n) DETACH DELETE n")
            # TODO more efficient:
            # USING PERIODIC COMMIT num_rows
            # MATCH (n)-[r]->(m)
            # DELETE r;
            # USING PERIODIC COMMIT num_rows
            # MATCH (n)
            # DETACH DELETE n;

    async def vector_search(
        self,
        label: str,
        prop_type: str,
        prop_name: str,
        embedding: np.ndarray | list[float],
        threshold: float,
        top_n: int = 5,
    ) -> list[Record]:
        with tracer.start_as_current_span("memgraph.vector_search") as span:
            span.set_attribute("memgraph.label", label)
            span.set_attribute("memgraph.prop_type", prop_type)
            span.set_attribute("memgraph.prop_name", prop_name)
            span.set_attribute("memgraph.top_n", top_n)
            span.set_attribute("memgraph.threshold", threshold)

            if prop_type == "edge":
                query = (
                    "CALL vector_search.search_edges($idx_name, $top_n, $embedding) "
                    "YIELD similarity, edge "
                    "WITH similarity, edge "
                    "WHERE similarity > $threshold "
                    "RETURN similarity, edge, startNode(edge) AS start, endNode(edge) AS end;"
                )
            else:  # node
                query = (
                    "CALL vector_search.search($idx_name, $top_n, $embedding) "
                    "YIELD similarity, node "
                    "WITH similarity, node WHERE similarity > $threshold RETURN *"
                )

            span.set_attribute("memgraph.query", query)

            idx_name = f"{label}_{prop_name}_embed_idx"
            res = await self.driver.execute_query(
                query,
                idx_name=idx_name,
                top_n=top_n,
                embedding=embedding,
                threshold=threshold,
                routing_=RoutingControl.READ,
            )
            return res.records

    async def create_vector_search_index(self, label: str, prop_name: str, dimension: int) -> None:
        idx_name = escape_cypher(f"{label}_{prop_name}_embed_idx")
        param_name = escape_cypher(f"{prop_name}_embedding")
        print(
            f"CREATE VECTOR INDEX {idx_name} ON :{label}({param_name}) "
            f'WITH CONFIG {{"dimension": {int(dimension)}, "capacity": 1024, "metric": "cos"}}'
        )
        # todo estimate vector index capacity from data
        await self.run_cypher(
            f"CREATE VECTOR INDEX {idx_name} ON :{label}({param_name}) "
            f'WITH CONFIG {{"dimension": {int(dimension)}, "capacity": 1024, "metric": "cos"}}',
        )

    async def text_search(self, label: str, query: str, limit: int = 10) -> Iterable[Record]:
        with tracer.start_as_current_span("memgraph.text_search") as span:
            span.set_attribute("memgraph.label", label)
            span.set_attribute("memgraph.fts_query", query)
            span.set_attribute("memgraph.limit", limit)

            query = "CALL text_search.search_all($idx_name, $query) YIELD node RETURN node LIMIT $limit"
            span.set_attribute("memgraph.query", query)

            res = await self.driver.execute_query(
                query,
                idx_name=self._fts_idx_name(label),
                query=query,
                limit=limit,
                routing_=RoutingControl.READ,
            )
            return res.records

    async def create_full_text_search_index(self, label: str) -> None:
        await self.run_cypher(
            "CREATE TEXT INDEX {idx_name} ON :{label}".format(
                label=escape_cypher(label),
                idx_name=escape_cypher(self._fts_idx_name(label)),
            )
        )

    async def create_node_prop_index(self, labels: set[str], property: str, unique: bool = False) -> None:
        escaped_label = escape_labels(labels)
        escaped_prop = escape_cypher(property)
        await self.run_cypher(f"CREATE INDEX ON :{escaped_label}({escaped_prop})")
        if not unique:
            return
        await self.run_cypher(f"CREATE CONSTRAINT ON (n:{escaped_label})\nASSERT n.{escaped_prop} IS UNIQUE")

    async def create_snapshot(self) -> None:
        await self.run_cypher("CREATE SNAPSHOT")

    @staticmethod
    def _fts_idx_name(label: str) -> str:
        return f"{label.lower()}_fts_idx"

    def close(self):
        self.driver.close()


class MemgraphGraphPgvectorVts(MemgraphGraph):
    """Cypher queries in Memgraph, vector search in Postgres+pgvector."""

    def __init__(
        self,
        uri: str,
        user: str,
        pwd: str,
        sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        super().__init__(uri=uri, user=user, pwd=pwd)
        self._sessionmaker: async_sessionmaker[AsyncSession] = sessionmaker or get_sessionmaker()

    async def vector_search(
        self,
        label: str,
        prop_type: str,
        prop_name: str,
        embedding: np.ndarray | list[float],
        threshold: float,
        top_n: int = 5,
    ) -> list[Record]:
        with tracer.start_as_current_span("pgvector.vector_search") as span:
            span.set_attribute("pgvector.label", label)
            span.set_attribute("pgvector.prop_type", prop_type)
            span.set_attribute("pgvector.prop_name", prop_name)
            span.set_attribute("pgvector.top_n", top_n)
            span.set_attribute("pgvector.threshold", threshold)

            if isinstance(embedding, np.ndarray):
                vec_list: list[float] = embedding.astype(float).tolist()
            else:
                vec_list = [float(x) for x in embedding]

            if prop_type == "edge":
                # Query edge embeddings from Postgres
                sa_query = """
                SELECT
                    edge_id,
                    from_node_id,
                    to_node_id,
                    edge_label,
                    1 - (embedding <=> (:vec)::vector) AS similarity
                FROM rag_edge_embeddings
                WHERE edge_label = :label
                AND attribute_name = :attr
                AND (1 - (embedding <=> (:vec)::vector)) > :threshold
                ORDER BY embedding <=> (:vec)::vector
                LIMIT :top_n;
                """
                span.set_attribute("pgvector.query", sa_query)

                async with self._sessionmaker() as session:
                    res = await session.execute(
                        text(sa_query),
                        {
                            "vec": vec_list,
                            "label": label,
                            "attr": prop_name,
                            "threshold": float(threshold),
                            "top_n": int(top_n),
                        },
                    )
                    rows = res.mappings().all()

                if not rows:
                    return []

                # Build mapping from edge tuple (from_id, to_id, edge_label) to similarity
                tuple_to_sim: dict[tuple[str, str, str], float] = {}  # (from_id, to_id, edge_label) -> similarity
                for r in rows:
                    from_id = str(r["from_node_id"])
                    to_id = str(r["to_node_id"])
                    edge_label = str(r["edge_label"])
                    similarity = float(r["similarity"])
                    tuple_to_sim[(from_id, to_id, edge_label)] = similarity

                # Fetch related data from Memgraph
                edge_params = [{"from_id": f, "to_id": t, "edge_label": el} for f, t, el in tuple_to_sim.keys()]
                cypher = f"""
                UNWIND $edges AS edge_tuple
                MATCH (from {{id: edge_tuple.from_id}})-[r:{escape_cypher(label)}]->(to {{id: edge_tuple.to_id}})
                RETURN from.id AS from_id, to.id AS to_id, '{label}' AS edge_label, r AS edge, from AS start, to AS end
                """
                mg_records = await self.execute_ro_cypher_query(cypher, {"edges": edge_params})

                # Build mapping from edge tuple to edge record
                tuple_to_edge: dict[tuple[str, str, str], dict[str, Any]] = {}
                for rec in mg_records:
                    rec_any = cast(neo4j.Record, rec)
                    from_id = str(rec_any["from_id"])
                    to_id = str(rec_any["to_id"])
                    edge_label = str(rec_any["edge_label"])
                    tuple_to_edge[(from_id, to_id, edge_label)] = {
                        "edge": rec_any["edge"],
                        "start": rec_any["start"],
                        "end": rec_any["end"],
                    }

                # Preserve similarity order and build output
                edge_results: list[Record] = []
                for (from_id, to_id, edge_label), sim in sorted(
                    tuple_to_sim.items(), key=lambda kv: kv[1], reverse=True
                ):
                    edge_data = tuple_to_edge.get((from_id, to_id, edge_label))
                    if edge_data is None:
                        continue
                    edge_results.append(
                        cast(
                            Record,
                            {
                                "similarity": sim,
                                "edge": edge_data["edge"],
                                "start": edge_data["start"],
                                "end": edge_data["end"],
                            },
                        )
                    )
                return edge_results

            else:  # node
                # Use SQLAlchemy sessionmaker (asyncpg driver) from vedana_core.db for consistency.
                sa_query = """
                SELECT
                    node_id,
                    1 - (embedding <=> (:vec)::vector) AS similarity
                FROM rag_anchor_embeddings
                WHERE label = :label
                AND attribute_name = :attr
                AND (1 - (embedding <=> (:vec)::vector)) > :threshold
                ORDER BY embedding <=> (:vec)::vector
                LIMIT :top_n;
                """
                span.set_attribute("pgvector.query", sa_query)

                async with self._sessionmaker() as session:
                    res = await session.execute(
                        text(sa_query),
                        {
                            "vec": vec_list,
                            "label": label,
                            "attr": prop_name,
                            "threshold": float(threshold),
                            "top_n": int(top_n),
                        },
                    )
                    rows = res.mappings().all()

                if not rows:
                    return []

                id_to_sim: dict[str, float] = {str(r["node_id"]): float(r["similarity"]) for r in rows}
                ids = list(id_to_sim.keys())

                # Fetch related data from Memgraph
                cypher = f"""
                MATCH (n:{escape_cypher(label)})
                WHERE n.id IN $ids
                RETURN n.id AS id, n AS node
                """
                mg_records = await self.execute_ro_cypher_query(cypher, {"ids": ids})

                id_to_node: dict[str, Any] = {}
                for rec in mg_records:
                    # rec is a neo4j.Record here
                    rec_any = cast(neo4j.Record, rec)
                    node_id = str(rec_any["id"])
                    id_to_node[node_id] = rec_any["node"]

                # Preserve similarity order
                node_results: list[Record] = []
                for node_id, sim in sorted(id_to_sim.items(), key=lambda kv: kv[1], reverse=True):
                    node = id_to_node.get(node_id)
                    if node is None:
                        continue
                    node_results.append(cast(Record, {"similarity": sim, "node": node}))
                return node_results

    def close(self):
        super().close()

    async def batch_add_nodes(self, label: str, node_dicts: list[dict]):
        cypher = f"""
        UNWIND $batch AS node
        CREATE (n:{escape_cypher(label)})
        SET n = node
        """
        await self.run_cypher(cypher, {"batch": node_dicts})

    async def batch_add_edges(self, label: str, edge_dicts: list[dict]):
        if not edge_dicts:
            return
        cypher = f"""
        UNWIND $batch AS edge
        MATCH (from {{id: edge.from_id}}), (to {{id: edge.to_id}})
        CREATE (from)-[r:{escape_cypher(label)}]->(to)
        SET r = edge
        """
        await self.run_cypher(cypher, {"batch": edge_dicts})


class Labels(set[str]): ...


class CypherQ:
    def __init__(self, q: te.LiteralString = "") -> None:
        self._q: str = q

    def _format_val(self, val: Any) -> str:
        if isinstance(val, CypherQ):
            return val._q
        if isinstance(val, Labels):
            return escape_labels(val)
        if isinstance(val, str):
            return escape_cypher(val)
        raise ValueError(f"Invalid type '{type(val).__name__}' passed to format()")

    def format(self, *args, **kwargs) -> "CypherQ":
        args = tuple(self._format_val(arg) for arg in args)
        kwargs = {k: self._format_val(v) for k, v in kwargs.items()}
        new_q = CypherQ()
        new_q._q = self._q.format(*args, **kwargs)
        return new_q

    def __str__(self) -> str:
        return self._q


def escape_cypher(identifier: str) -> str:
    identifier = identifier.replace("\u0060", "`").replace("`", "``")
    return f"`{identifier}`"


def escape_labels(labels: set[str]) -> str:
    return ":".join(escape_cypher(label) for label in labels)


_arg_name_re = re.compile(r"(\S+)")
_return_re = re.compile(r".+return\s+(.+?)(?:limit|order|where|$)", flags=re.IGNORECASE)


def extract_ret_attrs_names(cypher_query: str) -> list[str]:
    cypher_query = cypher_query.replace("\n", " ")
    return_stmts: list[str] = _return_re.findall(cypher_query)
    if not return_stmts:
        return []
    return_stmt = return_stmts[0]
    ret_args_stmts = return_stmt.split(",")
    args = []
    for arg_stmt in ret_args_stmts:
        candidates = _arg_name_re.findall(arg_stmt)
        if candidates:
            args.append(candidates[-1])
    return args


def main(): ...


if __name__ == "__main__":
    sys.exit(main())
