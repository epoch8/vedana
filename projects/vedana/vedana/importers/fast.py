import json
import logging
import multiprocessing
import time
from math import ceil
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union
from uuid import UUID

import numpy as np
from neo4j import GraphDatabase

from vedana.data_model import Anchor as DmAnchor
from vedana.data_model import DataModel
from vedana.data_model import Link as DmLink
from vedana.data_provider import Anchor, DataProvider
from vedana.embeddings import EmbeddingProvider
from vedana.graph import Graph, MemgraphGraph

logger = logging.getLogger(__name__)

DPIdType = tuple[str, int]


class BatchImporter:
    """
    High-performance data loader for Memgraph using parallel processing,
    analytical storage mode, and optimized Cypher patterns.
    """

    def __init__(
        self,
        graph: MemgraphGraph,
        dp: DataProvider,
        data_model: DataModel,
        embed_provider: EmbeddingProvider,
        node_batch_size: int = 1000,
        edge_batch_size: int = 3000,
        dry_run: bool = False,
    ):
        self.graph = graph
        self.dp = dp
        self.data_model = data_model
        self.embed_provider = embed_provider
        self.node_batch_size = node_batch_size
        self.edge_batch_size = edge_batch_size
        # self.num_processes = num_processes or max(1, multiprocessing.cpu_count() - 1)
        self.num_processes = 1
        self.dry_run = dry_run
        self.uri = graph.driver_uri
        self.auth = graph.auth
        self.anchor_embeddable_attributes = self._get_anchor_embeddable_attributes()
        self.link_embeddable_attributes = self._get_link_embeddable_attributes()
        self.embed_size = embed_provider.embeddings_dim
        # For foreign key link handling
        self.dp_to_graph_id_map: Dict[DPIdType, str] = {}
        self.f_key_links_to_add: List[Tuple[Union[str, DPIdType], Union[str, DPIdType], str]] = []

    # helper
    def is_uuid(self, val):
        try:
            UUID(val)
            return True
        except (ValueError, TypeError):
            return False

    def _get_anchor_embeddable_attributes(self) -> Dict[str, Set[str]]:
        return {
            anchor.noun: {attr.name for attr in anchor.attributes if attr.embeddable and attr.dtype == "str"}
            for anchor in self.data_model.anchors
        }

    def _get_link_embeddable_attributes(self) -> Dict[tuple[str, str], Set[str]]:
        return {
            (link.anchor_from.noun, link.anchor_to.noun): {
                attr.name for attr in link.attributes if attr.embeddable and attr.dtype == "str"
            }
            for link in self.data_model.links
        }

    def load_all_data(self) -> None:
        total_start_time = time.time()

        if not self.dry_run:
            self.graph.clear()
            logger.info("Graph cleared")

        # self.dp.get_anchor_types() - anchor tables provided in data source
        anchor_types = [a for a in self.data_model.anchors if a.noun in self.dp.get_anchor_types()]

        # Step 0: Create indices
        if not self.dry_run:
            logger.info("Creating indices")
            self.graph.create_basic_indices(node_types=[[e.noun] for e in anchor_types])

            logger.info("Creating vector indices")
            logger.info(f"node embeddable_attributes: {self.anchor_embeddable_attributes}")
            for label, attrs in self.anchor_embeddable_attributes.items():
                for attr in attrs:
                    # todo estimate vector index capacity
                    self.graph.create_vector_search_index(label, attr, self.embed_size)

            # Create vector indices for edge embeddable attributes
            logger.info(f"edge embeddable_attributes: {self.link_embeddable_attributes}")
            for (anchor_from, anchor_to), attrs in self.link_embeddable_attributes.items():
                for dm_link in self.data_model.links:
                    if dm_link.anchor_from.noun == anchor_from and dm_link.anchor_to.noun == anchor_to:
                        edge_type = dm_link.sentence
                        for attr in attrs:
                            self.graph.create_vector_search_index(edge_type, attr, self.embed_size)

        # Step 1: Load all nodes in parallel by type
        logger.info(f"Processing {len(anchor_types)} anchor types: {anchor_types}")

        for anchor_type in anchor_types:
            self._process_node_type(anchor_type)

        # Step 2.5: Process and insert foreign key links from anchor attributes
        if self.f_key_links_to_add:
            logger.info(f"Processing {len(self.f_key_links_to_add)} foreign key links")
            self._process_foreign_key_links()

        # Step 3: Load all edges in parallel by type
        link_types = self.dp.get_link_types()
        logger.info(f"Processing {len(link_types)} link types: {link_types}")

        for link_type in link_types:
            self._process_edge_type(link_type)

        # Step 4: Create snapshot
        if not self.dry_run:
            logger.info("Creating snapshot")
            self.graph.create_snapshot()

        total_time = time.time() - total_start_time
        logger.info(f"Total import completed in {total_time:.2f}s")

    def _handle_anchor_links(self, anchor: Anchor, dm_anchor_links: List[DmLink]) -> None:
        """Save anchor dp_id and foreign key links for future resolution"""
        if anchor.dp_id is not None:
            self.dp_to_graph_id_map[(anchor.type, anchor.dp_id)] = anchor.id

        for link in dm_anchor_links:
            linked_anchor_id: Optional[Union[str, int]] = None

            if link.anchor_from.noun == anchor.type and link.anchor_from_link_attr_name:
                linked_anchor_id = anchor.data.get(link.anchor_from_link_attr_name)
                linked_anchor_type = link.anchor_to.noun
                self._add_f_key_links(anchor, linked_anchor_type, linked_anchor_id, link.sentence, False)

            if link.anchor_to.noun == anchor.type and link.anchor_to_link_attr_name:
                linked_anchor_id = anchor.data.get(link.anchor_to_link_attr_name)
                linked_anchor_type = link.anchor_from.noun
                self._add_f_key_links(anchor, linked_anchor_type, linked_anchor_id, link.sentence, True)

    def _add_f_key_links(
        self,
        anchor: Anchor,
        linked_anchor_type: str,
        linked_anchor_id: Optional[Union[int, str]],
        label: str,
        reverse: bool,
    ) -> None:
        """Parse linked_anchor_id and save to f_key_links_to_add with correct direction"""
        if linked_anchor_id is None:
            return
        elif isinstance(linked_anchor_id, list):
            for link in linked_anchor_id:
                self._add_f_key_links(anchor, linked_anchor_type, link, label, reverse)
            return

        links_to_add: List[Tuple[str, DPIdType, str]] = []

        if isinstance(linked_anchor_id, int):
            links_to_add = [(anchor.id, (linked_anchor_type, linked_anchor_id), label)]
        elif isinstance(linked_anchor_id, str):
            try:
                ids = json.loads(linked_anchor_id)
                links_to_add = [(anchor.id, (linked_anchor_type, anchor_id), label) for anchor_id in ids]
            except ValueError:
                pass

        if reverse:
            links_to_add = [(anchor2, anchor1, label) for anchor1, anchor2, label in links_to_add]

        self.f_key_links_to_add.extend(links_to_add)

    def _process_node_type(self, anchor: DmAnchor) -> None:
        """Process all nodes of a specific type with parallel embedding calculation."""
        anchor_type = anchor.noun
        start_time = time.time()
        embeds_to_calc = self.anchor_embeddable_attributes.get(anchor_type) or set()

        # Get all anchors of this type
        anchors = self.dp.get_anchors(anchor_type, anchor.attributes)
        logger.info(f"Processing {len(anchors)} anchors of type {anchor_type}")

        if not anchors:
            return

        # Process anchor links in data model (foreign key relationships)
        dm_anchor_links = self.data_model.anchor_links(anchor_type)
        if dm_anchor_links:
            for a in anchors:
                self._handle_anchor_links(a, dm_anchor_links)

        # Calculate embeddings in batch, prepare nodes for upload
        anchor_attrs = [a.name for a in anchor.attributes]
        node_dicts, embedding_tasks = self._prepare_node_embedding_tasks(anchors, anchor_attrs, embeds_to_calc)

        if embedding_tasks:
            logger.info(f"Calculating {len(embedding_tasks)} embeddings for {anchor_type}")
            for ii, emebedding_batch in enumerate(self._split_into_batches(embedding_tasks, 500)):
                print(ii)
                embedding_results = self._process_embeddings(emebedding_batch)
                self._apply_embeddings_to_nodes(node_dicts, embedding_results)

        # Split nodes into batches for parallel processing
        batches = self._split_into_batches(node_dicts, self.node_batch_size)
        total_batches = ceil(len(node_dicts) / self.node_batch_size)

        if self.dry_run:
            logger.info(f"DRY RUN: Would insert {len(node_dicts)} nodes in {total_batches} batches")
        else:
            if self.num_processes > 1 and total_batches > 1:
                with multiprocessing.Pool(min(self.num_processes, total_batches)) as pool:
                    pool.starmap(
                        self._execute_node_batch,
                        [(anchor_type, batch, self.uri, self.auth) for batch in batches],
                    )
            else:
                for i, batch in enumerate(batches):
                    logger.debug(f"batch {i + 1} / {total_batches}")
                    self._execute_node_batch(anchor_type, batch, self.uri, self.auth)

        logger.info(f"Inserted {len(node_dicts)} nodes of type {anchor_type} in {time.time() - start_time:.2f}s")

    def _process_foreign_key_links(self) -> None:
        """Process and insert collected foreign key links in batches"""
        if not self.f_key_links_to_add or self.dry_run:
            return

        start_time = time.time()
        logger.info(f"Resolving and inserting {len(self.f_key_links_to_add)} foreign key links")

        # Resolve IDs
        resolved_links = []
        for link_from_id, link_to_id, label in self.f_key_links_to_add:
            try:
                from_id = self.dp_to_graph_id_map[link_from_id] if isinstance(link_from_id, tuple) else link_from_id
                to_id = self.dp_to_graph_id_map[link_to_id] if isinstance(link_to_id, tuple) else link_to_id
                resolved_links.append({"from_id": from_id, "to_id": to_id, "type": label})
            except KeyError:
                logger.error(f"Unable to map anchor dp id to graph id: {(link_from_id, link_to_id, label)}")
                continue

        # Group by type for batch processing
        links_by_type: Dict[str, List[Dict[str, str]]] = {}
        for link in resolved_links:
            links_by_type.setdefault(link["type"], []).append({"from_id": link["from_id"], "to_id": link["to_id"]})

        # Process each type in batches
        for edge_type, edges in links_by_type.items():
            batches = self._split_into_batches(edges, self.edge_batch_size)
            total_batches = ceil(len(edges) / self.edge_batch_size)
            logger.info(f"Processing {len(edges)} foreign key links of type '{edge_type}' in {total_batches} batches")

            if self.num_processes > 1 and total_batches > 1:
                with multiprocessing.Pool(min(self.num_processes, total_batches)) as pool:
                    pool.starmap(
                        self._execute_edge_batch,
                        [(edge_type, batch, self.uri, self.auth) for batch in batches],
                    )
            else:
                for batch in batches:
                    self._execute_edge_batch(edge_type, batch, self.uri, self.auth)

        logger.info(f"Inserted {len(resolved_links)} foreign key links in {time.time() - start_time:.2f}s")

    def _prepare_node_embedding_tasks(
        self, anchors: List[Anchor], anchor_attributes: list[str], embeds_to_calc: Set[str]
    ) -> Tuple[List[Dict[str, Any]], List[Tuple[str, str, str]]]:
        """Prepare nodes and collect text for embeddings."""
        node_dicts = []
        embedding_tasks = []  # [(node_id, attr_key, text)]

        for anchor in anchors:
            node = {
                "id": anchor.id,
                "type": anchor.type,
                **{k: v for k, v in anchor.data.items() if k in anchor_attributes},
            }
            node_dicts.append(node)

            # Collect texts that need embeddings
            for attr_key in embeds_to_calc:
                text = anchor.data.get(attr_key)
                if text and not self.is_uuid(text):
                    embedding_tasks.append((anchor.id, attr_key, text))

        return node_dicts, embedding_tasks

    def _process_embeddings(self, embedding_tasks: List[Tuple[str, str, str]]) -> Dict[Tuple[str, str], np.ndarray]:
        """Process embeddings in bulk using EmbeddingProvider."""
        texts = [task[2] for task in embedding_tasks]
        keys = [(task[0], task[1]) for task in embedding_tasks]

        embed_vecs = self.embed_provider.get_embeddings(texts)
        return {key: vec for key, vec in zip(keys, embed_vecs)}

    def _apply_embeddings_to_nodes(
        self,
        node_dicts: List[Dict[str, Any]],
        embedding_results: Dict[Tuple[str, str], np.ndarray],
    ) -> None:
        """Apply calculated embeddings to node dictionaries."""
        # Build a node_id -> node_dict map for efficient lookup
        node_map = {node["id"]: node for node in node_dicts}

        # Apply embeddings to respective nodes
        for (node_id, attr_key), embedding in embedding_results.items():
            if node_id in node_map:
                node_map[node_id][f"{attr_key}_embedding"] = embedding

    def _split_into_batches(self, items: List[Any], batch_size: int) -> Generator[List[Any], Any, None]:
        """Split a list into batches of specified size."""
        for i in range(0, len(items), batch_size):
            yield items[i : i + batch_size]

    @staticmethod
    def _execute_node_batch(node_type: str, batch: List[Dict[str, Any]], uri: str, auth: tuple[str, str]) -> None:
        """Execute a batch of node creations in a separate process."""
        driver = GraphDatabase.driver(uri, auth=auth)
        try:
            with driver.session() as session:
                query = f"""
                WITH $batch AS nodes
                UNWIND nodes AS node
                CREATE (n:{node_type})
                SET n = node
                """
                session.run(query, {"batch": batch})
        finally:
            driver.close()

    def _process_edge_type(self, link_type: str) -> None:
        start_time = time.time()

        links = self.dp.get_links(link_type)
        link_types = {(e.id_from.split(":")[0], e.id_to.split(":")[0]): e.type for e in links}
        logger.info(f"Processing {len(links)} links of type {link_type}")
        if not links:
            return

        # Get embeddable attributes for this link type
        embeds_to_calc = set()
        for (dm_from, dm_to), dm_sentence in link_types.items():
            attrs = self.link_embeddable_attributes.get((dm_from, dm_to))
            if attrs:
                embeds_to_calc.update(attrs)

        # Group edges by type (for case of directional edges this yields two keys)
        edge_dicts_by_type: Dict[str, List[Dict[str, Any]]] = {}
        edge_embedding_tasks: List[Tuple[str, str, str, str]] = []  # [(edge_id, edge_type, attr_key, text)]

        for link in links:
            if not link.id_from or not link.id_to:
                continue

            # Create unique edge identifier for embedding tracking
            edge_id = f"{link.id_from}-{link.type}-{link.id_to}"

            edge = {
                "from_id": link.id_from,
                "to_id": link.id_to,
                **link.data,
            }
            edge_dicts_by_type.setdefault(link.type, []).append(edge)

            # Collect texts that need embeddings for this edge
            for attr_key in embeds_to_calc:
                text = link.data.get(attr_key)
                if text:
                    edge_embedding_tasks.append((edge_id, link.type, attr_key, text))

        # Calculate embeddings for edges if needed
        if edge_embedding_tasks:
            logger.info(f"Calculating {len(edge_embedding_tasks)} edge embeddings for {link_type}")

            # Process embeddings in batches
            for ii, embedding_batch in enumerate(self._split_into_batches(edge_embedding_tasks, 500)):
                print(f"Processing edge embedding batch {ii}")
                embedding_results = self._process_edge_embeddings(embedding_batch)
                self._apply_embeddings_to_edges(edge_dicts_by_type, embedding_results)

        total_edges = sum(len(edges) for edges in edge_dicts_by_type.values())
        logger.info(f"Grouped {total_edges} valid edges into {len(edge_dicts_by_type)} types")

        if self.dry_run:
            logger.info(f"DRY RUN: Would insert {total_edges} edges")
            return

        for edge_type, edges in edge_dicts_by_type.items():
            batches = self._split_into_batches(edges, self.edge_batch_size)
            total_batches = ceil(len(edges) / self.edge_batch_size)
            logger.info(f"Processing {len(edges)} edges of type '{edge_type}' in {total_batches} batches")

            if self.num_processes > 1 and total_batches > 1:
                with multiprocessing.Pool(min(self.num_processes, total_batches)) as pool:
                    pool.starmap(
                        self._execute_edge_batch,
                        [(edge_type, batch, self.uri, self.auth) for batch in batches],
                    )
            else:
                for i, batch in enumerate(batches):
                    logger.debug(f"batch {i + 1}")
                    self._execute_edge_batch(edge_type, batch, self.uri, self.auth)

        logger.info(f"Inserted {total_edges} edges in {time.time() - start_time:.2f}s")

    def _process_edge_embeddings(
        self, embedding_tasks: List[Tuple[str, str, str, str]]
    ) -> Dict[Tuple[str, str, str], np.ndarray]:
        """Process edge embeddings in bulk using EmbeddingProvider."""
        texts = [task[3] for task in embedding_tasks]  # Extract text from (edge_id, edge_type, attr_key, text)
        keys = [(task[0], task[1], task[2]) for task in embedding_tasks]  # (edge_id, edge_type, attr_key)

        embed_vecs = self.embed_provider.get_embeddings(texts)
        return {key: vec for key, vec in zip(keys, embed_vecs)}

    def _apply_embeddings_to_edges(
        self,
        edge_dicts_by_type: Dict[str, List[Dict[str, Any]]],
        embedding_results: Dict[Tuple[str, str, str], np.ndarray],
    ) -> None:
        """Apply calculated embeddings to edge dictionaries."""
        # Build edge_id -> edge_dict map for efficient lookup across all types
        edge_map = {}
        for edge_type, edges in edge_dicts_by_type.items():
            for edge in edges:
                edge_id = f"{edge['from_id']}-{edge_type}-{edge['to_id']}"
                edge_map[edge_id] = edge

        # Apply embeddings to respective edges
        for (edge_id, edge_type, attr_key), embedding in embedding_results.items():
            if edge_id in edge_map:
                edge_map[edge_id][f"{attr_key}_embedding"] = embedding

    @staticmethod
    def _execute_edge_batch(edge_type: str, batch: List[Dict[str, Any]], uri: str, auth: tuple[str, str]) -> None:
        """Execute a batch of edge creations in a separate process."""
        driver = GraphDatabase.driver(uri, auth=auth)
        try:
            with driver.session() as session:
                query = f"""
                WITH $batch AS edges
                UNWIND edges AS edge
                MATCH (from {{id: edge.from_id}}), (to {{id: edge.to_id}})
                CREATE (from)-[r:{edge_type}]->(to)
                SET r = edge
                """
                session.run(query, {"batch": batch})
        finally:
            driver.close()


class DataModelLoader:
    def __init__(self, data_model: DataModel, graph: Graph) -> None:
        self.data_model = data_model
        self.graph = graph

    def update_data_model_node(self):
        """Persist current DataModel inside the target graph as a dedicated node.
        A single node with label `DataModel` and fixed id is used. The whole
        model is stored as JSON in the `content` property so that application
        instances can recreate the object without talking to Grist.
        """
        try:
            self.graph.run_cypher(
                "MERGE (dm:DataModel {id: 'data_model'}) SET dm.content = $content, dm.updated_at = datetime()",
                {"content": self.data_model.to_json()},
            )
            logger.info("DataModel node updated in graph")
        except Exception as exc:
            logger.error("Failed to update DataModel node: %s", exc)


def update_graph(
    graph: Graph,
    dp: DataProvider,
    data_model: DataModel,
    embed_provider: EmbeddingProvider,
    dry_run: bool = False,
    node_batch_size: int = 1000,
    edge_batch_size: int = 3000,
) -> None:
    """
    graph update function using parallel processing

    Args:
        graph: The graph database instance
        dp: Data provider with nodes and edges
        data_model: Data model for the graph
        embed_provider: Provider for generating embeddings
        dry_run: If True, don't actually write to the database
        node_batch_size: Size of node batches for parallel processing
        edge_batch_size: Size of edge batches for parallel processing
    """
    if not isinstance(graph, MemgraphGraph):
        raise NotImplementedError("update_graph only supports MemgraphGraph")

    loader = BatchImporter(
        graph=graph,
        dp=dp,
        data_model=data_model,
        embed_provider=embed_provider,
        node_batch_size=node_batch_size,
        edge_batch_size=edge_batch_size,
        dry_run=dry_run,
    )

    loader.load_all_data()
