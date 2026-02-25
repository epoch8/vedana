"""
Integration test: edge_attribute_filtering

Description:
  Only attributes described in Data Model -> Attributes should be included
  in edge attributes (rows where column `link` matches the link label
  from DM Links: `sentence`).

Data:
  Link document <-> regulation.
  Test data in Link_document_covers_regulation has an extra key
  `edge_attribute_extra`, which should NOT appear in the graph.

Checks:
  1) Find the record for document -> regulation in DM Links and get its `sentence`
     as the edge label.
  2) Collect allowed attribute keys for this link from DM Attributes
     (all `attribute_name` where `link == sentence`).
  3) Build the graph via steps.get_grist_data() and filter edges between document and regulation with the required label.
  4) Union of attributes keys across these edges:
     - does not contain 'edge_attribute_extra';
     - is a subset of the allowed set. If DM has no attributes for this link,
       edges should have no attributes at all.
"""

from typing import Dict, Set

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_edge_attribute_filtering() -> None:
    # 1) Data Model: find link document -> regulation and its sentence
    _anchors_df, a_attrs_df, l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not links_df.empty and not l_attrs_df.empty

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()
    dm_row = dm[(a1 == "document") & (a2 == "regulation")]
    assert not dm_row.empty, "DM has no link document -> regulation"

    sentence = str(dm_row.iloc[0]["sentence"]).strip()
    assert sentence

    # 2) Allowed edge attributes from DM (Attributes.link == sentence)
    if "link" in l_attrs_df.columns:
        mask = l_attrs_df["link"].astype(str).str.lower().str.strip() == sentence.lower()
    else:
        mask = pd.Series(False, index=l_attrs_df.index)

    allowed_edge_attrs: Set[str] = set(map(str, l_attrs_df.loc[mask, "attribute_name"].astype(str).tolist()))

    # 3) Data -> filter edges by DM
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty and not edges_df.empty

    # 4) Keep only document <-> regulation with the required label
    ft = edges_df["from_node_type"].astype(str).str.lower().str.strip()
    tt = edges_df["to_node_type"].astype(str).str.lower().str.strip()
    lbl = edges_df["edge_label"].astype(str).str.lower().str.strip()

    er = edges_df[
        (((ft == "document") & (tt == "regulation")) | ((ft == "regulation") & (tt == "document")))
        & (lbl == sentence.lower())
    ].copy()
    assert not er.empty, f"No edges '{sentence}' between document and regulation"

    # 5) Collect union of attribute keys across the found edges
    union_keys: Set[str] = set()
    has_any_attrs = False
    for _, row in er.iterrows():
        attrs: Dict[str, object] = row.get("attributes") or {}
        if attrs:
            has_any_attrs = True
            union_keys.update(map(str, attrs.keys()))

    # 5.1 Extra key from test data should not be present
    assert (
        "edge_attribute_extra" not in union_keys
    ), "Found unexpected edge attribute 'edge_attribute_extra' not present in Data Model."

    # 5.2 All keys from edges should be a subset of DM description
    if not allowed_edge_attrs:
        assert not has_any_attrs, f"DM has no edge attributes for '{sentence}', but edges carry: {sorted(union_keys)}"
    else:
        assert union_keys.issubset(
            allowed_edge_attrs
        ), f"Edge attributes not described in DM: {sorted(union_keys - allowed_edge_attrs)}"
