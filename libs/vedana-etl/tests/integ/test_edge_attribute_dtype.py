"""
Integration test: edge_attribute_dtype

Description:
For link DOCUMENT_covers_REGULATION the provider returns attribute values
with expected Python types:
- edge_attribute              -> str
- edge_attribute_integer      -> int
- edge_attribute_float        -> float
- edge_attribute_boolean      -> bool

Data:
Link: DOCUMENT_covers_REGULATION (document -> regulation)
Attribute: edge_attribute
We expect to find edges with this label having edge_attribute of different types.
"""

from typing import Dict, Set, Tuple, Type

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_edge_attribute_dtype() -> None:
    # 1) Get exact label from Data Model
    anchors_df, a_attrs_df, l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not links_df.empty, "Data Model Links is empty."

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()
    snt = dm["sentence"].astype(str)

    row = dm[(a1 == "document") & (a2 == "regulation") & (snt.str.lower().str.strip() == "document_covers_regulation")]
    assert not row.empty, "No document -> regulation record with sentence 'DOCUMENT_covers_REGULATION' in DM."

    sentence = str(row.iloc[0]["sentence"]).strip()

    # 2) Get raw edges without additional filtering
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes from Grist."
    assert isinstance(edges_df, pd.DataFrame) and not edges_df.empty, "edges_df is empty."

    er = edges_df[edges_df["edge_label"].astype(str).str.lower().str.strip() == sentence.lower()].copy()
    assert not er.empty, f"No edges with label '{sentence}'."

    # 3) Expected types by key
    expected_types: Dict[str, Type] = {
        "edge_attribute": str,
        "edge_attribute_integer": int,
        "edge_attribute_float": float,
        "edge_attribute_boolean": bool,
    }

    # track whether examples of each type are found
    seen: Dict[str, Tuple[object, str, str]] = {}

    # 4) Check types (note: bool is a subclass of int, so check bool before int)
    for _, r in er.iterrows():
        attrs: Dict[str, object] = r.get("attributes") or {}
        for key, typ in expected_types.items():
            if key not in attrs:
                continue
            val = attrs[key]
            # skip empty values
            if val is None:
                continue

            # strict type checking
            if typ is bool:
                assert isinstance(val, bool), f"{key} should be bool, got {type(val).__name__}: {val!r}"
            elif typ is int:
                # do not accept True/False as int
                assert isinstance(val, int) and not isinstance(
                    val, bool
                ), f"{key} should be int (not bool), got {type(val).__name__}: {val!r}"
            elif typ is float:
                assert isinstance(val, float), f"{key} should be float, got {type(val).__name__}: {val!r}"
            elif typ is str:
                assert isinstance(val, str), f"{key} should be str, got {type(val).__name__}: {val!r}"

            # remember at least one example for each key
            seen.setdefault(key, (val, str(r["from_node_id"]), str(r["to_node_id"])))

    # 5) Ensure at least one valid example was found for each key
    missing: Set[str] = set(expected_types) - set(seen)
    assert not missing, f"No values of expected type found for keys: {sorted(missing)}. Found examples: {seen}"
