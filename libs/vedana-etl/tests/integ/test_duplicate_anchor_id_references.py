"""
Интеграционный тест: duplicate_anchor_id_references

Описание:
При наличии дубликатов одного и того же логического узла (например, два ряда
Anchor_document с id "document:1"), в граф должен попасть только один узел
(дедуп по node_id), а любые Reference/Reference List, указывавшие на «дубликаты»,
должны быть распознаны как ссылки на тот узел, который остался в графе.

Данные:
- В Anchor_document есть дубликаты "document:1".
- В Anchor_document_chunk узлы "document_chunk:02/03/05" ссылаются на разные
  записи "document:1" (на уровне DP-ID), но в графе связи должны быть
  с единым узлом node_id == "document:1".

Проверяем:
1) В nodes существует РОВНО один узел с node_id == "document:1".
2) Между "document:1" и каждым из {"document_chunk:02","document_chunk:03","document_chunk:05"}
существует хотя бы одно ребро (направление неважно).
"""

from typing import Set

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def _has_edge_between(
    edges_df: pd.DataFrame,
    a_id: str,
    a_type: str,
    b_id: str,
    b_type: str,
) -> bool:
    """
    Проверка наличия ребра между (a_id, a_type) и (b_id, b_type) в любом направлении.
    """

    if edges_df.empty:
        return False
    from_id = edges_df["from_node_id"].astype(str)
    to_id = edges_df["to_node_id"].astype(str)
    from_t = edges_df["from_node_type"].astype(str).str.lower()
    to_t = edges_df["to_node_type"].astype(str).str.lower()

    mask_ab = (from_id == a_id) & (from_t == a_type.lower()) & (to_id == b_id) & (to_t == b_type.lower())
    mask_ba = (from_id == b_id) & (from_t == b_type.lower()) & (to_id == a_id) & (to_t == a_type.lower())
    return bool(edges_df[mask_ab | mask_ba].shape[0])


def test_duplicate_anchor_id_references() -> None:
    """
    Тест при дублировании ID узлов какие-либо Reference к дубликатам должны парситься
    как Reference к узлу, который окажется в графе.
    """

    # 1) Загружаем сырые graph-таблицы
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty, "Не получили узлы из Grist."
    assert isinstance(edges_df, pd.DataFrame), "edges_df должен быть DataFrame."

    # 2) Дедуп по "document:1": в графе должен остаться один узел с таким node_id
    doc1_rows = nodes_df[nodes_df["node_id"].astype(str) == "document:1"]
    assert not doc1_rows.empty, "В графе нет узла 'document:1'. Проверь тестовые данные."
    assert (
        doc1_rows.shape[0] == 1
    ), f"Ожидался ровно один узел 'document: 1' после дедупликации, получено {doc1_rows.shape[0]}."
    assert doc1_rows.iloc[0]["node_type"] == "document", "Узел 'document:1' должен быть типа 'document'."

    # 3) Межузловые связи: document:1 ↔ document_chunk:{02,03,05}
    required_chunks: Set[str] = {"document_chunk:02", "document_chunk:03", "document_chunk:05"}
    missing = [
        ch for ch in required_chunks if not _has_edge_between(edges_df, "document:1", "document", ch, "document_chunk")
    ]
    assert not missing, "Ожидались связи между 'document:1' и указанными чанками, но не найдены: " + ", ".join(
        sorted(missing)
    )
