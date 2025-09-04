"""
Интеграционный тест: edge_attribute_filtering

Описание:
  В граф у рёбер должны попадать только те атрибуты, которые описаны
  в Data Model → Attributes (строки, где column `link` соответствует
  метке связи из DM Links: `sentence`).

Данные:
  Связь document <-> regulation.
  В тестовых данных в Link_document_covers_regulation есть лишний ключ
  `edge_attribute_extra`, который НЕ должен оказаться в графе.

Проверяем:
  1) Находим в DM Links запись для document -> regulation и берём её `sentence`
     как метку ребра (edge_label).
  2) Собираем допустимые ключи атрибутов для этой связи из DM Attributes
     (все `attribute_name`, где `link == sentence`).
  3) Строим граф через steps.get_grist_data(), прогоняем steps.filter_grist_edges() и отбираем рёбра между
     document и regulation с нужной меткой.
  4) Объединение ключей attributes по этим рёбрам:
     - не содержит 'edge_attribute_extra';
     - является подмножеством допустимого набора. Если в DM для связи нет
       ни одного атрибута — у рёбер не должно быть атрибутов вовсе.
"""

from typing import Dict, Set

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_edge_attribute_filtering() -> None:
    # 1) Data Model: найдём link document -> regulation и его sentence
    anchors_df, attrs_df, links_df = next(steps.get_data_model())
    assert not links_df.empty and not attrs_df.empty

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()
    dm_row = dm[(a1 == "document") & (a2 == "regulation")]
    assert not dm_row.empty, "DM has no link document -> regulation"

    sentence = str(dm_row.iloc[0]["sentence"]).strip()
    assert sentence

    # 2) Разрешённые edge-атрибуты по DM (Attributes.link == sentence)
    if "link" in attrs_df.columns:
        mask = attrs_df["link"].astype(str).str.lower().str.strip() == sentence.lower()
    else:
        mask = pd.Series(False, index=attrs_df.index)

    allowed_edge_attrs: Set[str] = set(
        map(str, attrs_df.loc[mask, "attribute_name"].astype(str).tolist())
    )

    # 3) Данные → фильтрация рёбер по DM
    nodes_df, edges_df_before_filter = next(steps.get_grist_data())
    assert not nodes_df.empty and not edges_df_before_filter.empty

    # ВАЖНО: фильтруем рёбра по DM links
    edges_df = steps.filter_grist_edges(edges_df_before_filter, dm_links=links_df)

    # 4) Оставляем только document <-> regulation с нужной меткой
    ft = edges_df["from_node_type"].astype(str).str.lower().str.strip()
    tt = edges_df["to_node_type"].astype(str).str.lower().str.strip()
    lbl = edges_df["edge_label"].astype(str).str.lower().str.strip()

    er = edges_df[
        (
            ((ft == "document") & (tt == "regulation")) |
            ((ft == "regulation") & (tt == "document"))
        ) & (lbl == sentence.lower())
    ].copy()
    assert not er.empty, f"No edges '{sentence}' between document and regulation"

    # 5) Собираем объединение ключей attributes по найденным рёбрам
    union_keys: Set[str] = set()
    has_any_attrs = False
    for _, row in er.iterrows():
        attrs: Dict[str, object] = row.get("attributes") or {}
        if attrs:
            has_any_attrs = True
            union_keys.update(map(str, attrs.keys()))

    # 5.1 Лишний ключ из тестовых данных не должен попадать
    assert "edge_attribute_extra" not in union_keys, (
        "Found unexpected edge attribute 'edge_attribute_extra' not present in Data Model."
    )

    # 5.2 Все ключи из рёбер должны быть подмножеством DM-описания
    if not allowed_edge_attrs:
        assert not has_any_attrs, (
            f"DM has no edge attributes for '{sentence}', but edges carry: {sorted(union_keys)}"
        )
    else:
        assert union_keys.issubset(allowed_edge_attrs), (
            "Edge attributes not described in DM: "
            f"{sorted(union_keys - allowed_edge_attrs)}"
        )