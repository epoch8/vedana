"""
Integration test: anchor_link_columns

Description:
Reference / Reference List columns in Anchor_* tables should generate edges between
nodes according to Data Model.

Data:
- link document <-> document_chunk is defined by a column in Anchor_document:
  link_document_has_document_chunk
- document 'document:1' in test data has references to chunks:
  document_chunk:01, document_chunk:02, document_chunk:03, document_chunk:05

Checks:
1) Find link document -> document_chunk in DM and get its sentence.
2) Get edges from pipeline (steps.get_grist_data()).
3) Verify that edges_df contains edges between 'document:1' and the listed chunks
   with the required edge_label (from DM). Direction doesn't matter.
"""

from typing import Set, Tuple

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def _unordered(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def test_anchor_link_columns() -> None:
    # 1) Get sentence from Data Model for document <-> document_chunk
    _anchors_df, _a_attrs_df, _l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not links_df.empty, "Data Model Links is empty."

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()
    row = dm[(a1 == "document") & (a2 == "document_chunk")]
    assert not row.empty, "No link document -> document_chunk in Data Model."
    sentence = str(row.iloc[0]["sentence"]).strip()
    assert sentence, "Empty sentence for document <-> document_chunk in Data Model."

    # 2) Get edges from pipeline
    nodes_df, edges_df = next(steps.get_grist_data())
    assert isinstance(edges_df, pd.DataFrame) and not edges_df.empty, "edges_df is empty."

    # Filter required edges: document <-> document_chunk, required label
    ft = edges_df["from_node_type"].astype(str).str.lower().str.strip()
    tt = edges_df["to_node_type"].astype(str).str.lower().str.strip()
    lbl = edges_df["edge_label"].astype(str).str.lower().str.strip()

    target_edges = edges_df[
        (((ft == "document") & (tt == "document_chunk")) | ((ft == "document_chunk") & (tt == "document")))
        & (lbl == sentence.lower())
    ].copy()

    assert not target_edges.empty, f"No edges document <-> document_chunk with label '{sentence}' found."

    # Build set of actual undirected pairs
    actual_pairs: Set[Tuple[str, str]] = set(
        _unordered(str(r["from_node_id"]).strip(), str(r["to_node_id"]).strip()) for _, r in target_edges.iterrows()
    )

    # 3) Expected pairs for 'document:1' from test data
    expected_pairs: Set[Tuple[str, str]] = {
        _unordered("document:1", "document_chunk:01"),
        _unordered("document:1", "document_chunk:02"),
        _unordered("document:1", "document_chunk:03"),
        _unordered("document:1", "document_chunk:05"),
    }

    missing = sorted(p for p in expected_pairs if p not in actual_pairs)
    assert not missing, f"Not all links from Anchor_document reference column are in the graph. Missing pairs: {missing}"
