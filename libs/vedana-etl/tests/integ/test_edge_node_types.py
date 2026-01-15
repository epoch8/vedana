"""
Интеграционный тест: edge_node_types

Описание:
Проверить, что типы узлов from_node_type / to_node_type у рёбер берутся
из Data Model (или корректно выводятся по node_id), даже если строки в
таблице Link_* записаны не в «правильном» порядке.

Данные:
Link_regulation_is_described_in_document содержит 2 строки:
  1) reg:001 -> document:1
  2) document:002 -> reg:002     (поля местами)

Проверяем:
Мы ожидаем в графе рёбра для обеих пар в обоих направлениях с корректными типами:
  - reg:001 -> document:1   (from_type=regulation, to_type=document)
  - document:1 -> reg:001   (реверс для недиректной связи)
  - reg:002 -> document:002 (несмотря на «перевёрнутую» запись в Link_*)
  - document:002 -> reg:002 (реверс)
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
    # 1) Берём из Data Model точный sentence и убеждаемся, что связь недиректная
    anchors_df, a_attrs_df, l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    assert not links_df.empty

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()

    row = dm[(a1 == "regulation") & (a2 == "document")]
    assert not row.empty, "В DM нет линки regulation -> document."

    sentence = str(row.iloc[0]["sentence"]).strip()
    has_direction = bool(row.iloc[0].get("has_direction", False))
    # В этом кейсе линк должен быть недиректным, чтобы в графе было 2 направления
    assert not has_direction, "Ожидалась недиректная связь для теста bidirectional."

    # 2) Забираем рёбра из пайплайна (get_grist_data уже добавляет реверсы и нормализует типы)
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty and not edges_df.empty

    # 3) Проверяем обе пары ID и оба направления
    #    Пара 1: reg:001 <-> document:1
    assert _has_edge(
        edges_df, "reg:001", "document:1", "regulation", "document", sentence
    ), "Нет ребра regulation->document для пары reg:001/document:1."
    assert _has_edge(
        edges_df, "document:1", "reg:001", "document", "regulation", sentence
    ), "Нет обратного ребра document->regulation для пары reg:001/document:1."

    #    Пара 2: документ в Link_* был «from», но в графе должен появиться и нормализованный reg->document
    assert _has_edge(
        edges_df, "reg:002", "document:002", "regulation", "document", sentence
    ), "Нет нормализованного ребра regulation->document для пары reg:002/document:002."
    assert _has_edge(
        edges_df, "document:002", "reg:002", "document", "regulation", sentence
    ), "Нет обратного ребра document->regulation для пары reg:002/document:002."
