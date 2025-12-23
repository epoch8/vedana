import abc
import logging
from typing import Sequence

import numpy as np
from neo4j import AsyncGraphDatabase, RoutingControl, Record
from opentelemetry import trace
from sqlalchemy import select, RowMapping
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from vedana_etl.catalog import rag_anchor_embeddings, rag_edge_embeddings, nodes, edges

from vedana_core.db import get_sessionmaker

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class VectorStore(abc.ABC):
    async def vector_search(
            self,
            label: str,
            prop_type: str,
            prop_name: str,
            embedding: list[float],
            threshold: float,
            top_n: int = 5,
    ):
        raise NotImplementedError


class MemgraphVectorStore(VectorStore):
    """
    Use Memgraph vector_search capabilities. Requires vector indices to be created separately.
    """
    def __init__(self, uri: str, user: str, pwd: str, db_name: str = "") -> None:
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, pwd), database=db_name)
        # await self.driver.verify_connectivity()
        self.driver_uri = uri
        self.auth = (user, pwd)

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


class PGVectorStore(VectorStore):
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    ) -> None:
        self._sessionmaker: async_sessionmaker[AsyncSession] = sessionmaker or get_sessionmaker()
        self.rag_anchor_embeddings_table = rag_anchor_embeddings.store.data_table  # type: ignore[attr-defined]
        self.rag_edge_embeddings_table = rag_edge_embeddings.store.data_table  # type: ignore[attr-defined]
        self.node_table = nodes.store.data_table  # type: ignore[attr-defined]
        self.edge_table = edges.store.data_table  # type: ignore[attr-defined]

    async def vector_search(
        self,
        label: str,
        prop_type: str,
        prop_name: str,
        embedding: np.ndarray | list[float],
        threshold: float,
        top_n: int = 5,
    ) -> Sequence[RowMapping]:
        with tracer.start_as_current_span("pgvector.vector_search") as span:
            span.set_attribute("pgvector.label", label)
            span.set_attribute("pgvector.prop_type", prop_type)
            span.set_attribute("pgvector.prop_name", prop_name)
            span.set_attribute("pgvector.top_n", top_n)
            span.set_attribute("pgvector.threshold", threshold)

            async with self._sessionmaker() as session:
                if prop_type == "edge":
                    distance = self.rag_edge_embeddings_table.c.embedding.cosine_distance(embedding)
                    similarity = (1 - distance).label("similarity")

                    stmt = (
                        select(
                            similarity,
                            self.edge_table.c.from_node_id,
                            self.edge_table.c.to_node_id,
                            self.edge_table.c.edge_label,
                            self.edge_table.c.attributes.label("edge"),
                        )
                        .select_from(
                            self.rag_edge_embeddings_table.join(
                                self.edge_table,
                                (self.rag_edge_embeddings_table.c.from_node_id == self.edge_table.c.from_node_id)
                                & (self.rag_edge_embeddings_table.c.to_node_id == self.edge_table.c.to_node_id)
                                & (self.rag_edge_embeddings_table.c.edge_label == self.edge_table.c.edge_label)
                            )
                        )
                        .where(self.rag_edge_embeddings_table.c.edge_label == label)
                        .where(self.rag_edge_embeddings_table.c.attribute_name == prop_name)
                        .where(similarity > threshold)
                        .order_by(distance)
                        .limit(top_n)
                    )

                else:  # node
                    distance = self.rag_anchor_embeddings_table.c.embedding.cosine_distance(embedding)
                    similarity = (1 - distance).label("similarity")

                    stmt = (
                        select(
                            similarity,
                            self.node_table.c.node_id,
                            self.node_table.c.node_type,
                            self.node_table.c.attributes.label("node"),
                        )
                        .select_from(
                            self.rag_anchor_embeddings_table.join(
                                self.node_table,
                                self.rag_anchor_embeddings_table.c.node_id == self.node_table.c.node_id
                            )
                        )
                        .where(self.rag_anchor_embeddings_table.c.label == label)
                        .where(self.rag_anchor_embeddings_table.c.attribute_name == prop_name)
                        .where(similarity > threshold)
                        .order_by(distance)
                        .limit(top_n)
                    )

                span.set_attribute("pgvector.query", str(stmt))
                res = await session.execute(stmt)
                return res.mappings().all()
