"""
Integration test: duplicate_anchor_ids

Description:
  When node IDs are duplicated, the pipeline should not break.
  The first described node for a given node_id should be included in the graph.

Test data:
  Anchor_document has two objects with the same node_id "document:1".
  After drop_duplicates by "node_id", the first record is expected to remain
  (in the test Grist snapshot this is the document with document_name = "doc_a").

Checks:
  1) The final nodes_df contains exactly one node with node_id == "document:1".
  2) Its attributes correspond to the first record (document_name == "doc_a").
"""

from typing import Dict

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_duplicate_anchor_ids_keep_first() -> None:
    """
    Verify that when duplicates by node_id exist, the first record is kept.
    """

    # 1) Load graph tables from live Grist via standard pipeline step.
    nodes_df, _ = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes fetched from Grist; check test data and GRIST_* env."

    # 2) Filter by the problematic identifier from test data.
    masked = nodes_df[nodes_df["node_id"] == "document:1"]
    assert not masked.empty, "Expected at least one node with node_id == 'document:1' in test data."

    # 3) Duplicates should be collapsed: exactly one row remains.
    assert len(masked) == 1, f"Duplicate node_id 'document: 1' wasn't deduplicated. Found {len(masked)} rows."

    # 4) Verify that the first record was kept (expected name 'doc_a').
    attrs: Dict[str, object] = masked.iloc[0]["attributes"] or {}
    got_name = attrs.get("document_name")
    assert got_name == "doc_a", f"""
        Deduplication didn't preserve the first record for 'document:1'.
        Expected document_name == 'doc_a', got {got_name!r}.
        """
