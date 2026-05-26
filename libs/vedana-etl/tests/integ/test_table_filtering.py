"""
Integration test: table_filtering

Goal:
  Only tables with Anchor_/Link_ prefixes should be loaded into the pipeline.
  Practically this means:
    - Only node types listed in Data Model (Anchors) are included in nodes.
    - Any other tables (e.g., meta_document_reference_attrs) should NOT be loaded into the pipeline.

Data:
  The test Grist has a service table `meta_document_reference_attrs`, which should NOT be loaded into the pipeline.

Checks:
  1) The set of unique node_type from get_grist_data() (except the service 'DataModel')
  is a subset of {noun from Anchors}.
  2) node_type does not contain values starting with 'meta_' (case-insensitive).
"""

from typing import Set

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_table_filtering() -> None:
    """
    Verify that only Anchor_/Link_ tables are loaded into the graph, and meta_* tables are not loaded.
    """

    # 1) Load Data Model (Anchors/Attributes/Links) and raw nodes/edges from live Grist.
    anchors_df, a_attrs_df, l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    nodes_df, edges_df = next(steps.get_grist_data())

    # Check for data presence in Grist.
    assert not anchors_df.empty, "Anchors in Data Model must not be empty."
    assert not nodes_df.empty, "No nodes fetched from Grist."

    # 2) Allowed node types are only those listed in the Anchors notation of Data Model.
    allowed_node_types: Set[str] = set(anchors_df["noun"].astype(str))

    # 'DataModel' is always present - exclude it from filtering check.
    actual_node_types: Set[str] = set(nodes_df.loc[nodes_df["node_type"] != "DataModel", "node_type"].astype(str))

    # 2.1) All node types from data must correspond to Data Model anchors.
    assert actual_node_types.issubset(allowed_node_types), f"""
        Found node types that do not correspond to Data Model Anchors (allowed={sorted(allowed_node_types)},
        actual={sorted(actual_node_types)}, extra={sorted(actual_node_types - allowed_node_types)})
        """

    # 3) Additionally ensure that meta-table did not turn into nodes.
    # Check that there are no types starting with 'meta_' (case-insensitive).
    lower_types = {t.lower() for t in actual_node_types}
    banned_prefix = "meta_"
    offending = sorted(t for t in lower_types if t.startswith(banned_prefix))
    assert not offending, f"Meta-table leaked into nodes: {offending}"

    # Ensure at least one "normal" type from Data Model is present.
    assert any(t in lower_types for t in ("document", "document_chunk", "regulation")), """
        Expected at least one of typical anchor types ('document', 'document_chunk', 'regulation')
        to be present in the nodes. Adjust this assertion if your DM differs.
        """
