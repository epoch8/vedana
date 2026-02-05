"""
Integration test: anchor_attribute_filtering

What we check:
  - ONLY attributes described in Data Model are included in the graph.
  - Specifically verify that "document_random_attr" (present in test data)
    is completely removed from "document" type nodes.

Steps:
  1) Load Data Model (Anchors/Attributes/Links) from live Grist.
  2) Load raw data from live Grist.
  3) Verify:
     - attributes (except service DataModel) are a subset of Data Model attributes,
       only allowing *_embedding (may appear later).
     - for "document" type, key "document_random_attr" is absent.
"""

from typing import Dict, Set

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_anchor_attribute_filtering_removes_unknown() -> None:
    """
    Main verification of node attribute filtering.
    """

    # 1) Verify Data Model
    anchors_df, a_attrs_df, _l_attrs_df, _links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not anchors_df.empty and not a_attrs_df.empty, "Data Model must not be empty (Anchors)."

    # 2) Verify data from Grist
    nodes_df, _ = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes fetched from Grist."

    # 3) Allowed attributes from Data Model
    allowed_attrs: Set[str] = set(a_attrs_df["attribute_name"].astype(str))

    # 3.1) For each node (except DataModel) attribute keys should be subset of Data Model attributes (plus *_embedding)
    for _, row in nodes_df[nodes_df["node_type"] != "DataModel"].iterrows():
        attr_dict: Dict[str, object] = row["attributes"] or {}
        keys = set(map(str, attr_dict.keys()))
        # allow generated embeddings that may appear later
        embedding_keys = {k for k in keys if k.endswith("_embedding")}
        unknown = keys - allowed_attrs - embedding_keys
        assert not unknown, f"""
            Node {row["node_id"]} ({row["node_type"]}) contains attributes not described in
            Data Model: {sorted(unknown)}
            """

    # 3.2) Special case: document_random_attr should be removed from document nodes
    docs = nodes_df[nodes_df["node_type"] == "document"]
    assert not docs.empty, "Expected at least one 'document' node in test data."
    still_has_random = [
        row["node_id"] for _, row in docs.iterrows() if "document_random_attr" in (row["attributes"] or {})
    ]
    assert (
        not still_has_random
    ), f"Unexpected attribute 'document_random_attr' is still present in document nodes: {still_has_random}"

    # Ensure document nodes still have at least one valid field from DM, so the test doesn't pass
    # with an "empty" attribute set.
    any_valid_left = any(bool(set((row["attributes"] or {}).keys()) & allowed_attrs) for _, row in docs.iterrows())
    assert any_valid_left, "After filtering, document nodes should still have at least one attribute from Data Model."
