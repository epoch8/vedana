"""
Интеграционный тест: edge_attribute_dtype

Описание:
Для связи DOCUMENT_covers_REGULATION провайдер возвращает значения атрибутов
с ожидаемыми Python-типами:
- edge_attribute              -> str
- edge_attribute_integer      -> int
- edge_attribute_float        -> float
- edge_attribute_boolean      -> bool

Данные:
Связь: DOCUMENT_covers_REGULATION (document -> regulation)
Атрибут: edge_attribute
Ожидаем, что среди рёбер с этим label найдутся экземпляры с edge_attribute разных типов.
"""

from typing import Dict, Set, Tuple, Type

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_edge_attribute_dtype() -> None:
    # 1) Точный label берём из Data Model
    anchors_df, attrs_df, links_df = next(steps.get_data_model())
    assert not links_df.empty, "Data Model Links пуст."

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()
    snt = dm["sentence"].astype(str)

    row = dm[
        (a1 == "document")
        & (a2 == "regulation")
        & (snt.str.lower().str.strip() == "document_covers_regulation")
    ]
    assert not row.empty, "В DM нет записи document -> regulation с sentence 'DOCUMENT_covers_REGULATION'."

    sentence = str(row.iloc[0]["sentence"]).strip()

    # 2) Берём сырые рёбра без доп. фильтраций
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty, "Нет узлов из Grist."
    assert isinstance(edges_df, pd.DataFrame) and not edges_df.empty, "edges_df пуст."

    er = edges_df[edges_df["edge_label"].astype(str).str.lower().str.strip() == sentence.lower()].copy()
    assert not er.empty, f"Нет рёбер с label '{sentence}'."

    # 3) Ожидаемые типы по ключам
    expected_types: Dict[str, Type] = {
        "edge_attribute": str,
        "edge_attribute_integer": int,
        "edge_attribute_float": float,
        "edge_attribute_boolean": bool,
    }

    # найдены ли примеры каждого типа
    seen: Dict[str, Tuple[object, str, str]] = {}

    # 4) Проверяем типы (учтём, что bool — подкласс int, поэтому проверяем раньше int)
    for _, r in er.iterrows():
        attrs: Dict[str, object] = r.get("attributes") or {}
        for key, typ in expected_types.items():
            if key not in attrs:
                continue
            val = attrs[key]
            # пропускаем пустышки
            if val is None:
                continue

            # строгая проверка типов
            if typ is bool:
                assert isinstance(val, bool), (
                    f"{key} должен быть bool, а получен {type(val).__name__}: {val!r}"
                )
            elif typ is int:
                # не принимаем True/False как int
                assert isinstance(val, int) and not isinstance(val, bool), (
                    f"{key} должен быть int (не bool), а получен {type(val).__name__}: {val!r}"
                )
            elif typ is float:
                assert isinstance(val, float), (
                    f"{key} должен быть float, а получен {type(val).__name__}: {val!r}"
                )
            elif typ is str:
                assert isinstance(val, str), (
                    f"{key} должен быть str, а получен {type(val).__name__}: {val!r}"
                )

            # запоминаем хотя бы один пример для каждого ключа
            seen.setdefault(key, (val, str(r["from_node_id"]), str(r["to_node_id"])))

    # 5) Убедимся, что по каждому ключу нашёлся хотя бы один валидный пример
    missing: Set[str] = set(expected_types) - set(seen)
    assert not missing, (
        "Не найдены значения ожидаемого типа по ключам: "
        f"{sorted(missing)}. Найденные примеры: {seen}"
    )
