"""
Integration test: edge_bidirectional

Description:
If a link in Data Model has has_direction == False, then the pipeline (get_grist_data)
should generate TWO edges for each node pair: anchor1->anchor2 and anchor1<-anchor2,
and their attributes should be identical (usually {} for references from Anchor_*).

Data:
DOCUMENT_has_DOCUMENT_CHUNK (document <-> document_chunk)
"""

from typing import Dict, Tuple

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def _norm(s: object) -> str:
    return str(s).strip().lower()


def _as_attr_dict(val: object) -> Dict[str, object]:
    if isinstance(val, dict):
        return val
    if pd.isna(val):
        return {}

    # just in case: the provider should not put strings here
    return dict(val) if isinstance(val, dict) else {}


def test_edge_bidirectional() -> None:
    # 1) Get the required link from Data Model and verify it's non-directional
    anchors_df, a_attrs_df, l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not links_df.empty, "Data Model Links is empty."

    dm = links_df.copy()
    row = dm[
        (dm["anchor1"].astype(str).str.lower().str.strip() == "document")
        & (dm["anchor2"].astype(str).str.lower().str.strip() == "document_chunk")
        & (dm["sentence"].astype(str).str.len() > 0)
    ]
    assert not row.empty, "No document <-> document_chunk link in DM."
    sentence = str(row.iloc[0]["sentence"]).strip()
    has_direction = bool(row.iloc[0].get("has_direction", False))
    assert not has_direction, f"Link '{sentence}' in DM is marked as directed, expected bidirectional."

    # 2) Get edges from pipeline (get_grist_data adds reverse-edge for non-directional links)
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty and not edges_df.empty, "Empty nodes/edges from Grist."

    # 3) Filter only our document <-> document_chunk link
    lbl = edges_df["edge_label"].astype(str).str.lower().str.strip()
    ft = edges_df["from_node_type"].astype(str).str.lower().str.strip()
    tt = edges_df["to_node_type"].astype(str).str.lower().str.strip()

    is_our_type = ((ft == "document") & (tt == "document_chunk")) | ((ft == "document_chunk") & (tt == "document"))
    er = edges_df[(lbl == sentence.lower()) & is_our_type].copy()
    assert not er.empty, f"No edges found for link '{sentence}'."

    # 4) Group by undirected node pair and verify each pair has two directions
    # pair key: (min_id, max_id)
    def undirected_key(row: pd.Series) -> Tuple[str, str]:
        a = str(row["from_node_id"]).strip()
        b = str(row["to_node_id"]).strip()
        return (a, b) if a <= b else (b, a)

    er["pair_key"] = er.apply(undirected_key, axis=1)

    # Additional check: there is at least one unique pair in the data
    pairs = er["pair_key"].unique().tolist()
    assert pairs, "No unique node pairs found for bidirectional link."

    for pair in pairs:
        sub = er[er["pair_key"] == pair]

        # There should be exactly 2 edges: A->B and B->A
        assert sub.shape[0] == 2, f"For pair {pair} expected 2 edges (both directions), found {sub.shape[0]}.\n{sub}"

        a_to_b = sub.iloc[0]
        b_to_a = sub.iloc[1]

        # Verify they are indeed opposite directions
        assert (
            str(a_to_b["from_node_id"]).strip() == str(b_to_a["to_node_id"]).strip()
        ), f"Expected opposite directions for pair {pair}."
        assert (
            str(a_to_b["to_node_id"]).strip() == str(b_to_a["from_node_id"]).strip()
        ), f"Expected opposite directions for pair {pair}."

        # Node types should also be swapped
        assert _norm(a_to_b["from_node_type"]) == _norm(b_to_a["to_node_type"])
        assert _norm(a_to_b["to_node_type"]) == _norm(b_to_a["from_node_type"])

        # Attributes should match (usually empty dict for references from Anchor_*),
        # but we compare strictly.
        attrs1 = _as_attr_dict(a_to_b.get("attributes"))
        attrs2 = _as_attr_dict(b_to_a.get("attributes"))
        assert attrs1 == attrs2, f"Attributes of opposite edges differ for pair {pair}: {attrs1} vs {attrs2}"
