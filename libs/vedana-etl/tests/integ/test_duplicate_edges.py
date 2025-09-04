"""
Интеграционный тест: duplicate_edges

Описание:
Дубликаты связей между узлами должны игнорироваться: в итоговом графе
на каждую упорядоченную пару (from_node_id, to_node_id) с данным edge_label
остаётся ровно 1 запись.

Данные:
- документ `document:1` связан с `reg:001` через Link_document_covers_regulation,
  и такая связь продублирована (например, одновременно через Link_* и через FK-колонку Anchor_*).

Проверяем:
1) В DM находим link document -> regulation и берём его `sentence` как edge_label.
2) Среди рёбер document <-> regulation с этим label нет дублей по
   (from_node_id, to_node_id, edge_label).
3) Для пары document:1 ↔ reg:001 с этим label в графе осталась ровно одна связь
   (в любом направлении — допускаем A->B или B->A).
"""

from typing import List, Tuple

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def _group_offenders_by_ordered_pair(edges: pd.Series) -> List[Tuple[str, str, str]]:
    """
    Вернуть список упорядоченных пар (from_id, to_id, label), для которых найдено >1 записи.
    """

    if edges.empty:
        return []
    grouped = (
        edges.assign(
            f=edges["from_node_id"].astype(str),
            t=edges["to_node_id"].astype(str),
            l=edges["edge_label"].astype(str),
        )[["f", "t", "l"]]
        .value_counts()
    )
    return [(fr, to, label) for (fr, to, label), cnt in grouped.items() if cnt > 1]


def test_duplicate_edges() -> None:
    # 1) Data Model: достаём метку нужного линка
    anchors_df, attrs_df, links_df = next(steps.get_data_model())
    assert not links_df.empty, "Data Model Links пуст."

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()
    row = dm[(a1 == "document") & (a2 == "regulation")]
    assert not row.empty, "В DM нет link document -> regulation."

    sentence = str(row.iloc[0]["sentence"]).strip()
    assert sentence, "Пустая метка (sentence) у link document -> regulation в DM."

    # 2) Граф из Grist
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty, "Нет узлов из Grist."
    assert isinstance(edges_df, pd.DataFrame) and not edges_df.empty, "edges_df пуст."

    # 3) Фильтруем рёбра по типам и label
    ft = edges_df["from_node_type"].astype(str).str.lower().str.strip()
    tt = edges_df["to_node_type"].astype(str).str.lower().str.strip()
    lbl = edges_df["edge_label"].astype(str).str.lower().str.strip()

    er = edges_df[
        (
            ((ft == "document") & (tt == "regulation")) |
            ((ft == "regulation") & (tt == "document"))
        )
        & (lbl == sentence.lower())
    ].copy()
    assert not er.empty, f"Нет рёбер '{sentence}' между document и regulation."

    # 4) Глобально: не должно быть дублей по упорядоченным парам (from, to, label)
    offenders = _group_offenders_by_ordered_pair(er)
    assert not offenders, (
        "Обнаружены дубликаты связей (ожидалась 1 запись на упорядоченную пару from->to с данным label): "
        + ", ".join([f"{from_id} -> {to_id} [{label}]" for from_id, to_id, label in offenders])
    )

    # 5) Точно для пары document:1 ↔ reg:001 — должна существовать ровно одна связь (в любом направлении)
    er_doc1_reg1 = er[
        ((er["from_node_id"].astype(str) == "document:1") & (er["to_node_id"].astype(str) == "reg:001")) |
        ((er["from_node_id"].astype(str) == "reg:001") & (er["to_node_id"].astype(str) == "document:1"))
    ]
    assert not er_doc1_reg1.empty, (
        f"Ожидалась хотя бы одна связь '{sentence}' между 'document: 1' и 'reg: 001'."
    )
    assert er_doc1_reg1.shape[0] == 1, (
        f"Ожидалась ровно одна связь для пары 'document: 1' ↔ 'reg: 001' с label '{sentence}', "
        f"получено {er_doc1_reg1.shape[0]}."
    )
