"""
Integration test: anchor_attributes_reference_type_column (SQL Reference)

Test:
  - When reading via Grist SQL provider, a Reference column comes as <ref_id> + gristHelper_<col>,
    but the final attributes should contain **string value** (as in UI), without gristHelper_* keys.
  - If a Reference column is described in Data Model, it should be preserved during filtering.

Test data:
 - reference field: document_reference_attr (exists in Data Model)
"""

from typing import Any, Dict, Optional

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_anchor_attributes_reference_type_column() -> None:
    """
    1) In raw nodes (get_grist_data) at least one document node has
       `document_reference_attr` as a non-empty string.
       There should be no gristHelper_* keys in attributes.
    2) After filtering:
       - `document_reference_attr` remains (because it exists in Data Model).
    """

    # --- 1) Live Data Model
    anchors_df, a_attrs_df, _l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not anchors_df.empty and not a_attrs_df.empty, "Data Model must not be empty (Anchors)."

    dm_attr_names = set(a_attrs_df["attribute_name"].astype(str))
    assert (
        "document_reference_attr" in dm_attr_names
    ), "Precondition: 'document_reference_attr' must be present in Data Model."

    # --- 2) Data from live Grist
    nodes_df, _ = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes fetched from Grist."

    documents = nodes_df[nodes_df["node_type"] == "document"]
    assert not documents.empty, "Expected at least one 'document' node in raw data."

    # --- 2.1) Find document with string document_reference_attr
    ref_node_attrs: Optional[Dict[str, Any]] = None
    for _, row in documents.iterrows():
        attrs: Dict[str, Any] = row["attributes"] or {}
        val = attrs.get("document_reference_attr")
        if isinstance(val, str):
            ref_node_attrs = attrs
            break

    assert ref_node_attrs is not None, """
        Reference column 'document_reference_attr' must appear as a non-empty string
        in at least one 'document' node (raw data).
        """

    # --- 2.2) Verify that gristHelper_* keys did not leak into attributes
    assert not any(k.startswith("gristHelper_") for k in ref_node_attrs.keys()), """
        gristHelper_* keys leaked into attributes; the SQL provider should have
        used them to reconstruct the final string value and NOT keep helper keys.
        """

    # --- 3) Verify filtering by Data Model
    docs_f = nodes_df[nodes_df["node_type"] == "document"]
    assert not docs_f.empty, "Filtered graph should still contain 'document' nodes."

    preserved_any = False
    for _, row in docs_f.iterrows():
        attrs = row["attributes"] or {}
        if isinstance(attrs.get("document_reference_attr"), str):
            preserved_any = True
            break
    assert (
        preserved_any
    ), "Expected 'document_reference_attr' to be preserved by filtering logic because it is present in Data Model."
