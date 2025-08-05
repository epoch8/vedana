import abc
import json
import logging
import re
import sys
from itertools import islice
from typing import Any, Dict, Iterable, Set

import neo4j
import numpy as np
import typing_extensions as te
from neo4j import EagerResult, GraphDatabase, RoutingControl
from opentelemetry import trace

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

Record = neo4j.Record


class Graph(abc.ABC):
    @abc.abstractmethod
    def add_node(
        self,
        node_id: str,
        labels: Set[str],
        properties: dict[str, Any] | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> None: ...

    @abc.abstractmethod
    def add_edge(self, from_id: str, to_id: str, type_: str, attrs: Dict[str, Any] | None) -> None: ...

    @abc.abstractmethod
    def number_of_nodes(self) -> int: ...

    @abc.abstractmethod
    def number_of_edges(self) -> int: ...

    @abc.abstractmethod
    def run_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterable[Record]: ...

    def get_existing_node_types(self) -> Iterable[list[str]]:
        raise NotImplementedError

    def create_full_text_search_index(self, label: str) -> None:
        raise NotImplementedError

    def create_vector_search_index(self, label: str, prop_name: str, dimension: int) -> None:
        raise NotImplementedError

    def create_snapshot(self) -> None:
        raise NotImplementedError

    def llm_schema(self) -> str:
        raise NotImplementedError

    def text_search(self, label: str, query: str, limit: int = 10) -> Iterable[Record]:
        raise NotImplementedError

    def vector_search(
        self,
        label: str,
        prop_name: str,
        embedding: np.ndarray | list[float],
        threshold: float,
        top_n: int = 10,
    ) -> list[Record]:
        raise NotImplementedError

    def setup(self, *_, create_basic_indices: bool = True, **kwargs) -> None:
        # Set false to speedup import
        if create_basic_indices:
            self.create_basic_indices()

    def create_basic_indices(self) -> None: ...

    def execute_ro_cypher_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> Iterable[Record]:
        return self.run_cypher(query, parameters, limit=limit)

    def clear(self) -> None: ...

    def close(self) -> None: ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CypherGraph(Graph):
    def add_node(
        self,
        node_id: str,
        labels: Set[str],
        properties: dict[str, Any] | None = None,
        embeddings: dict[str, np.ndarray] | None = None,
    ) -> None:
        query, params = self._add_node_cypher(node_id, labels, properties or {})
        self.run_cypher(query, params)

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

    def add_edge(self, from_id: str, to_id: str, type_: str, attrs: Dict[str, Any] | None) -> None:
        query, params = self._add_edge_cypher(from_id, to_id, type_, attrs)
        self.run_cypher(query, params)

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

    def add_edges(self, edges: Iterable[tuple[str, str, dict]], **common_attrs) -> None:
        for edge_tuple in edges:
            from_id, to_id, attrs = edge_tuple
            attrs = {**common_attrs, **attrs}
            labels: Iterable[str] = attrs.pop("__labels__", [])
            type_ = next(iter(labels), "no_type")
            self.add_edge(from_id, to_id, type_, attrs)

    def number_of_nodes(self) -> int:
        res = self.execute_ro_cypher_query("MATCH (n) RETURN count(*) as cnt")
        return next(iter(res))["cnt"]

    def number_of_edges(self) -> int:
        res = self.execute_ro_cypher_query("MATCH (f)-[]->(t) RETURN count(*) as cnt")
        return next(iter(res))["cnt"]

    def get_existing_node_types(self) -> Iterable[list[str]]:
        res = self.execute_ro_cypher_query("MATCH (n) RETURN DISTINCT labels(n) as l;")
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
        self.driver = GraphDatabase.driver(uri, auth=(user, pwd), database=db_name)
        self.driver.verify_connectivity()
        self.driver_uri = uri
        self.auth = (user, pwd)

    def execute_ro_cypher_query(
        self, query: str, parameters: dict[str, Any] | None = None, limit: int | None = None
    ) -> Iterable[Record]:
        with tracer.start_as_current_span("memgraph.execute_ro_cypher_query") as span:
            span.set_attribute("memgraph.query", query)
            if parameters:
                span.set_attribute("memgraph.parameters", json.dumps(parameters))
            result: EagerResult = self.driver.execute_query(query, parameters, routing_=RoutingControl.READ)

        return result.records

    def run_cypher(
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

            with self.driver.session() as session:
                result = list(islice(session.run(query, parameters), limit))

        return result

    def add_node(
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
        super().add_node(node_id, labels, properties, embeddings)

    def llm_schema(self) -> str:
        res = self.driver.execute_query("CALL llm_util.schema() YIELD schema RETURN schema")
        return res.records[0]["schema"]

    def create_basic_indices(self, node_types=None) -> None:
        if not node_types:
            node_types = self.get_existing_node_types()
        for label in node_types:
            self.create_node_prop_index(set(label), "id", unique=True)

    def clear(self):
        with self.driver.session() as session:
            for (idx_name,) in session.run("CALL vector_search.show_index_info() YIELD index_name RETURN *"):
                session.run(f"DROP VECTOR INDEX {escape_cypher(idx_name)}")
            idx_name_re = re.compile(r"\(name:\s(.+?)\)")
            for row in session.run("SHOW INDEX INFO"):
                index_type = row["index type"]
                idx_name = next(iter(idx_name_re.findall(index_type)), None)
                if not idx_name:
                    continue
                session.run(f"DROP TEXT INDEX {escape_cypher(idx_name)}")
            session.run("CALL schema.assert({}, {}, {}, true) YIELD action, key, keys, label, unique")
            session.run("MATCH (n) DETACH DELETE n")
            # TODO more efficient:
            # USING PERIODIC COMMIT num_rows
            # MATCH (n)-[r]->(m)
            # DELETE r;
            # USING PERIODIC COMMIT num_rows
            # MATCH (n)
            # DETACH DELETE n;

    def vector_search(
        self,
        label: str,
        prop_name: str,
        embedding: np.ndarray | list[float],
        threshold: float,
        top_n: int = 5,
    ) -> list[Record]:
        with tracer.start_as_current_span("memgraph.vector_search") as span:
            span.set_attribute("memgraph.label", label)
            span.set_attribute("memgraph.prop_name", prop_name)
            span.set_attribute("memgraph.top_n", top_n)
            span.set_attribute("memgraph.threshold", threshold)

            query = (
                "CALL vector_search.search($idx_name, $top_n, $embedding) YIELD similarity, node "
                "WITH similarity, node WHERE similarity > $threshold RETURN *"
            )
            span.set_attribute("memgraph.query", query)

            idx_name = f"{label}_{prop_name}_embed_idx"
            res = self.driver.execute_query(
                query,
                idx_name=idx_name,
                top_n=top_n,
                embedding=embedding,
                threshold=threshold,
                routing_=RoutingControl.READ,
            )
            return res.records

    def create_vector_search_index(self, label: str, prop_name: str, dimension: int) -> None:
        idx_name = escape_cypher(f"{label}_{prop_name}_embed_idx")
        param_name = escape_cypher(f"{prop_name}_embedding")
        print(
            f"CREATE VECTOR INDEX {idx_name} ON :{label}({param_name}) "
            f'WITH CONFIG {{"dimension": {int(dimension)}, "capacity": 1024, "metric": "cos"}}'
        )
        # todo estimate vector index capacity from data
        self.run_cypher(
            f"CREATE VECTOR INDEX {idx_name} ON :{label}({param_name}) "
            f'WITH CONFIG {{"dimension": {int(dimension)}, "capacity": 1024, "metric": "cos"}}',
        )

    def text_search(self, label: str, query: str, limit: int = 10) -> Iterable[Record]:
        with tracer.start_as_current_span("memgraph.text_search") as span:
            span.set_attribute("memgraph.label", label)
            span.set_attribute("memgraph.fts_query", query)
            span.set_attribute("memgraph.limit", limit)

            query = "CALL text_search.search_all($idx_name, $query) YIELD node RETURN node LIMIT $limit"
            span.set_attribute("memgraph.query", query)

            res = self.driver.execute_query(
                query,
                idx_name=self._fts_idx_name(label),
                query=query,
                limit=limit,
                routing_=RoutingControl.READ,
            )
            return res.records

    def create_full_text_search_index(self, label: str) -> None:
        self.run_cypher(
            "CREATE TEXT INDEX {idx_name} ON :{label}".format(
                label=escape_cypher(label),
                idx_name=escape_cypher(self._fts_idx_name(label)),
            )
        )

    def create_node_prop_index(self, labels: set[str], property: str, unique: bool = False) -> None:
        escaped_label = escape_labels(labels)
        escaped_prop = escape_cypher(property)
        self.run_cypher(f"CREATE INDEX ON :{escaped_label}({escaped_prop})")
        if not unique:
            return
        self.run_cypher(f"CREATE CONSTRAINT ON (n:{escaped_label})\nASSERT n.{escaped_prop} IS UNIQUE")

    def create_snapshot(self) -> None:
        self.run_cypher("CREATE SNAPSHOT")

    @staticmethod
    def _fts_idx_name(label: str) -> str:
        return f"{label.lower()}_fts_idx"

    def close(self):
        self.driver.close()

    def batch_add_nodes(self, label: str, node_dicts: list[dict]):
        cypher = f"""
        UNWIND $batch AS node
        CREATE (n:{escape_cypher(label)})
        SET n = node
        """
        self.run_cypher(cypher, {"batch": node_dicts})

    def batch_add_edges(self, label: str, edge_dicts: list[dict]):
        if not edge_dicts:
            return
        cypher = f"""
        UNWIND $batch AS edge
        MATCH (from {{id: edge.from_id}}), (to {{id: edge.to_id}})
        CREATE (from)-[r:{escape_cypher(label)}]->(to)
        SET r = edge
        """
        self.run_cypher(cypher, {"batch": edge_dicts})


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
