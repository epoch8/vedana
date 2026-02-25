"""
Integration test: edge_node_types

Description:
Verify that node types from_node_type / to_node_type on edges are taken
from Data Model (or correctly inferred from node_id), even if rows in
Link_* table are written in the "wrong" order.

Data:
Link_regulation_is_described_in_document contains 2 rows:
  1) reg:001 -> document:1
  2) document:002 -> reg:002     (fields swapped)

Checks:
We expect edges in the graph for both pairs in both directions with correct types:
  - reg:001 -> document:1   (from_type=regulation, to_type=document)
  - document:1 -> reg:001   (reverse for non-directional link)
  - reg:002 -> document:002 (despite the "reversed" record in Link_*)
  - document:002 -> reg:002 (reverse)
"""

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def _has_edge(
    df: pd.DataFrame,
    from_id: str,
    to_id: str,
    from_type: str,
    to_type: str,
    sentence: str,
) -> bool:
    f = df["from_node_id"].astype(str).str.strip()
    t = df["to_node_id"].astype(str).str.strip()
    ft = df["from_node_type"].astype(str).str.lower().str.strip()
    tt = df["to_node_type"].astype(str).str.lower().str.strip()
    lbl = df["edge_label"].astype(str).str.lower().str.strip()

    mask = (
        (f == from_id) & (t == to_id) & (ft == from_type.lower()) & (tt == to_type.lower()) & (lbl == sentence.lower())
    )
    return bool(df[mask].shape[0])


def test_edge_node_types() -> None:
    # 1) Get the exact sentence from Data Model and verify the link is non-directional
    anchors_df, a_attrs_df, l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not links_df.empty

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()

    row = dm[(a1 == "regulation") & (a2 == "document")]
    assert not row.empty, "No regulation -> document link in DM."

    sentence = str(row.iloc[0]["sentence"]).strip()
    has_direction = bool(row.iloc[0].get("has_direction", False))
    # In this case the link should be non-directional for the graph to have 2 directions
    assert not has_direction, "Expected non-directional link for bidirectional test."

    # 2) Get edges from pipeline (get_grist_data already adds reverses and normalizes types)
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty and not edges_df.empty

    # 3) Check both ID pairs and both directions
    #    Pair 1: reg:001 <-> document:1
    assert _has_edge(
        edges_df, "reg:001", "document:1", "regulation", "document", sentence
    ), "No regulation->document edge for pair reg:001/document:1."
    assert _has_edge(
        edges_df, "document:1", "reg:001", "document", "regulation", sentence
    ), "No reverse document->regulation edge for pair reg:001/document:1."

    #    Pair 2: document was "from" in Link_*, but normalized reg->document should appear in the graph
    assert _has_edge(
        edges_df, "reg:002", "document:002", "regulation", "document", sentence
    ), "No normalized regulation->document edge for pair reg:002/document:002."
    assert _has_edge(
        edges_df, "document:002", "reg:002", "document", "regulation", sentence
    ), "No reverse document->regulation edge for pair reg:002/document:002."
