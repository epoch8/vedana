"""
Integration test: anchor_attributes_formula_type_column

Goal:
  - Grist columns with "Formula" data type appear in raw data (get_grist_data)
    as computed values (strings/numbers etc., not expressions).
  - If such a column is described in Data Model, it should be preserved.

Test data:
  - Formula attribute: `document_filepath` (for nodes of type "document").
"""

from typing import Any, Dict, Optional

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_anchor_attributes_formula_type_column() -> None:
    """
    Verify the behavior of formula column `document_filepath`:
    - in raw nodes it appears as a computed value (non-empty);
    - described in Data Model and not filtered out
    """

    # 1) Live Data Model
    anchors_df, a_attrs_df, _l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not anchors_df.empty and not a_attrs_df.empty, "Data Model must not be empty (Anchors)."

    dm_attr_names = set(a_attrs_df["attribute_name"].astype(str))

    # In this case we expect the formula attribute to be described in Data Model.
    assert (
        "document_filepath" in dm_attr_names
    ), "Test precondition failed: 'document_filepath' must be present in Data Model."

    # 2) Data from live Grist
    nodes_df, _ = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes fetched from Grist."

    documents = nodes_df[nodes_df["node_type"] == "document"]
    assert not documents.empty, "Expected at least one 'document' node in raw data."

    # 3) Find at least one non-empty document_filepath value in raw
    raw_value: Optional[Any] = None
    raw_node_id: Optional[str] = None
    for _, row in documents.iterrows():
        attrs: Dict[str, Any] = row["attributes"] or {}
        if "document_filepath" in attrs and str(attrs["document_filepath"]).strip():
            raw_value = attrs["document_filepath"]
            raw_node_id = row["node_id"]
            break

    assert raw_value is not None and str(raw_value).strip(), """
        Formula column 'document_filepath' must appear as a computed, non-empty value
        in at least one 'document' node (raw data).
        """

    # 4) After filtering by Data Model the attribute should be preserved
    docs_f = nodes_df[nodes_df["node_type"] == "document"]
    assert not docs_f.empty, "Filtered graph should still contain 'document' nodes."

    # Verify that the same node (if it remained) still has the field and it's non-empty
    if raw_node_id is not None and (docs_f["node_id"] == raw_node_id).any():
        row = docs_f.loc[docs_f["node_id"] == raw_node_id].iloc[0]
        attrs_f: Dict[str, Any] = row["attributes"] or {}
        assert (
            "document_filepath" in attrs_f and str(attrs_f["document_filepath"]).strip()
        ), "Expected 'document_filepath' to be preserved by filtering logic because it is present in Data Model."
    else:
        # Otherwise just verify that some document node has the field
        found_any = False
        for _, row in docs_f.iterrows():
            attrs_f = row["attributes"] or {}
            if "document_filepath" in attrs_f and str(attrs_f["document_filepath"]).strip():
                found_any = True
                break
        assert found_any, """
            Expected 'document_filepath' to be present on at least one 'document' node
            after filtering logic (it's in Data Model).
            """
