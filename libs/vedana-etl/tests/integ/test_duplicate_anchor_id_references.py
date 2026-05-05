"""
Integration test: duplicate_anchor_id_references

Description:
When duplicates of the same logical node exist (e.g., two rows in
Anchor_document with id "document:1"), only one node should be included
in the graph (dedup by node_id), and any Reference/Reference List pointing
to "duplicates" should be recognized as references to the node that remained in the graph.

Data:
- Anchor_document has duplicates "document:1".
- In Anchor_document_chunk, nodes "document_chunk:02/03/05" reference different
  "document:1" records (at DP-ID level), but in the graph links should be
  with the single node node_id == "document:1".

Checks:
1) In nodes there exists EXACTLY one node with node_id == "document:1".
2) Between "document:1" and each of {"document_chunk:02","document_chunk:03","document_chunk:05"}
there exists at least one edge (direction doesn't matter).
"""

from typing import Set

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def _has_edge_between(
    edges_df: pd.DataFrame,
    a_id: str,
    a_type: str,
    b_id: str,
    b_type: str,
) -> bool:
    """
    Check for edge existence between (a_id, a_type) and (b_id, b_type) in any direction.
    """

    if edges_df.empty:
        return False
    from_id = edges_df["from_node_id"].astype(str)
    to_id = edges_df["to_node_id"].astype(str)
    from_t = edges_df["from_node_type"].astype(str).str.lower()
    to_t = edges_df["to_node_type"].astype(str).str.lower()

    mask_ab = (from_id == a_id) & (from_t == a_type.lower()) & (to_id == b_id) & (to_t == b_type.lower())
    mask_ba = (from_id == b_id) & (from_t == b_type.lower()) & (to_id == a_id) & (to_t == a_type.lower())
    return bool(edges_df[mask_ab | mask_ba].shape[0])


def test_duplicate_anchor_id_references() -> None:
    """
    Test that when node IDs are duplicated, any Reference to duplicates should be parsed
    as Reference to the node that ends up in the graph.
    """

    # 1) Load raw graph tables
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes received from Grist."
    assert isinstance(edges_df, pd.DataFrame), "edges_df should be a DataFrame."

    # 2) Dedup by "document:1": only one node with this node_id should remain in the graph
    doc1_rows = nodes_df[nodes_df["node_id"].astype(str) == "document:1"]
    assert not doc1_rows.empty, "No 'document:1' node in the graph. Check test data."
    assert (
        doc1_rows.shape[0] == 1
    ), f"Expected exactly one 'document: 1' node after deduplication, got {doc1_rows.shape[0]}."
    assert doc1_rows.iloc[0]["node_type"] == "document", "Node 'document:1' should be of type 'document'."

    # 3) Inter-node links: document:1 <-> document_chunk:{02,03,05}
    required_chunks: Set[str] = {"document_chunk:02", "document_chunk:03", "document_chunk:05"}
    missing = [
        ch for ch in required_chunks if not _has_edge_between(edges_df, "document:1", "document", ch, "document_chunk")
    ]
    assert not missing, "Expected links between 'document:1' and specified chunks, but not found: " + ", ".join(
        sorted(missing)
    )
