"""
Интеграционный тест: edge_bidirectional

Описание:
Если в Data Model у связи has_direction == False, то пайплайн (get_grist_data)
должен породить ДВА ребра для каждой пары узлов: anchor1->anchor2 и anchor1<-anchor2,
и их атрибуты должны быть идентичны (обычно это {} для ссылок из Anchor_*).

Данные:
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

    # на всякий случай: провайдер не должен сюда ставить строки
    return dict(val) if isinstance(val, dict) else {}


def test_edge_bidirectional() -> None:
    # 1) Берём из Data Model нужную ссылку и убеждаемся, что она недиректная
    anchors_df, attrs_df, links_df = next(steps.get_data_model())
    assert not links_df.empty, "Data Model Links пуст."

    dm = links_df.copy()
    row = dm[
        (dm["anchor1"].astype(str).str.lower().str.strip() == "document")
        & (dm["anchor2"].astype(str).str.lower().str.strip() == "document_chunk")
        & (dm["sentence"].astype(str).str.len() > 0)
    ]
    assert not row.empty, "В DM нет связи document <-> document_chunk."
    sentence = str(row.iloc[0]["sentence"]).strip()
    has_direction = bool(row.iloc[0].get("has_direction", False))
    assert not has_direction, f"Связь '{sentence}' в DM помечена как направленная, ожидалась двунаправленная."

    # 2) Забираем рёбра из пайплайна (get_grist_data добавляет reverse-ребро для недиректных связей)
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty and not edges_df.empty, "Пустые nodes/edges из Grist."

    # 3) Фильтруем только нашу связь document <-> document_chunk
    lbl = edges_df["edge_label"].astype(str).str.lower().str.strip()
    ft = edges_df["from_node_type"].astype(str).str.lower().str.strip()
    tt = edges_df["to_node_type"].astype(str).str.lower().str.strip()

    is_our_type = (
        ((ft == "document") & (tt == "document_chunk"))
        | ((ft == "document_chunk") & (tt == "document"))
    )
    er = edges_df[(lbl == sentence.lower()) & is_our_type].copy()
    assert not er.empty, f"Не найдено ни одного ребра для связи '{sentence}'."

    # 4) Группируем по неориентированной паре узлов и проверяем, что на каждую пару есть два направления
    # ключ пары: (min_id, max_id)
    def undirected_key(row: pd.Series) -> Tuple[str, str]:
        a = str(row["from_node_id"]).strip()
        b = str(row["to_node_id"]).strip()
        return (a, b) if a <= b else (b, a)

    er["pair_key"] = er.apply(undirected_key, axis=1)

    # Доп. проверка: в данных есть хотя бы одна уникальная пара
    pairs = er["pair_key"].unique().tolist()
    assert pairs, "Не найдено ни одной уникальной пары узлов для двунаправленной связи."

    for pair in pairs:
        sub = er[er["pair_key"] == pair]

        # Должно быть ровно 2 ребра: A->B и B->A
        assert sub.shape[0] == 2, (
            f"Для пары {pair} ожидаются 2 ребра (в обе стороны), найдено {sub.shape[0]}.\n{sub}"
        )

        a_to_b = sub.iloc[0]
        b_to_a = sub.iloc[1]

        # Проверим, что действительно противоположные направления
        assert str(a_to_b["from_node_id"]).strip() == str(b_to_a["to_node_id"]).strip(), (
            f"Ожидались противоположные направления у пары {pair}."
        )
        assert str(a_to_b["to_node_id"]).strip() == str(b_to_a["from_node_id"]).strip(), (
            f"Ожидались противоположные направления у пары {pair}."
        )

        # Типы узлов тоже должны меняться местами
        assert _norm(a_to_b["from_node_type"]) == _norm(b_to_a["to_node_type"])
        assert _norm(a_to_b["to_node_type"]) == _norm(b_to_a["from_node_type"])

        # Атрибуты должны совпадать (обычно это пустой dict для ссылок из Anchor_*),
        # но сравним строго.
        attrs1 = _as_attr_dict(a_to_b.get("attributes"))
        attrs2 = _as_attr_dict(b_to_a.get("attributes"))
        assert attrs1 == attrs2, (
            f"Атрибуты противоположных рёбер различаются для пары {pair}: {attrs1} vs {attrs2}"
        )
