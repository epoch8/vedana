"""
Integration test: duplicate_edges

Description:
Duplicate links between nodes should be ignored: in the final graph
for each ordered pair (from_node_id, to_node_id) with a given edge_label
exactly 1 record remains.

Data:
- document `document:1` is linked to `reg:001` via Link_document_covers_regulation,
  and this link is duplicated (e.g., simultaneously via Link_* and FK-column in Anchor_*).

Checks:
1) Find link document -> regulation in DM and get its `sentence` as edge_label.
2) Among edges document <-> regulation with this label there are no duplicates by
   (from_node_id, to_node_id, edge_label).
3) For pair document:1 <-> reg:001 with this label exactly one link remains in the graph
   (in any direction - either A->B or B->A is acceptable).
"""

from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def _group_offenders_by_ordered_pair(edges: pd.Series) -> List[Tuple[str, str, str]]:
    """
    Return a list of ordered pairs (from_id, to_id, label) for which >1 record was found.
    """

    if edges.empty:
        return []
    grouped = edges.assign(
        f=edges["from_node_id"].astype(str),
        t=edges["to_node_id"].astype(str),
        l=edges["edge_label"].astype(str),
    )[["f", "t", "l"]].value_counts()
    return [(fr, to, label) for (fr, to, label), cnt in grouped.items() if cnt > 1]


def test_duplicate_edges() -> None:
    # 1) Data Model: get the required link label
    anchors_df, a_attrs_df, l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not links_df.empty, "Data Model Links is empty."

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()
    row = dm[(a1 == "document") & (a2 == "regulation")]
    assert not row.empty, "No link document -> regulation in DM."

    sentence = str(row.iloc[0]["sentence"]).strip()
    assert sentence, "Empty label (sentence) for link document -> regulation in DM."

    # 2) Graph from Grist
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes from Grist."
    assert isinstance(edges_df, pd.DataFrame) and not edges_df.empty, "edges_df is empty."

    # 3) Filter edges by types and label
    ft = edges_df["from_node_type"].astype(str).str.lower().str.strip()
    tt = edges_df["to_node_type"].astype(str).str.lower().str.strip()
    lbl = edges_df["edge_label"].astype(str).str.lower().str.strip()

    er = edges_df[
        (((ft == "document") & (tt == "regulation")) | ((ft == "regulation") & (tt == "document")))
        & (lbl == sentence.lower())
    ].copy()
    assert not er.empty, f"No edges '{sentence}' between document and regulation."

    # 4) Globally: there should be no duplicates by ordered pairs (from, to, label)
    offenders = _group_offenders_by_ordered_pair(er)
    assert not offenders, (
        "Duplicate links detected (expected 1 record per ordered pair from->to with given label): "
        + ", ".join([f"{from_id} -> {to_id} [{label}]" for from_id, to_id, label in offenders])
    )

    # 5) Specifically for pair document:1 <-> reg:001 - exactly one link should exist (in any direction)
    er_doc1_reg1 = er[
        ((er["from_node_id"].astype(str) == "document:1") & (er["to_node_id"].astype(str) == "reg:001"))
        | ((er["from_node_id"].astype(str) == "reg:001") & (er["to_node_id"].astype(str) == "document:1"))
    ]
    assert not er_doc1_reg1.empty, f"Expected at least one link '{sentence}' between 'document: 1' and 'reg: 001'."
    assert er_doc1_reg1.shape[0] == 1, (
        f"Expected exactly one link for pair 'document: 1' <-> 'reg: 001' with label '{sentence}', "
        f"got {er_doc1_reg1.shape[0]}."
    )
