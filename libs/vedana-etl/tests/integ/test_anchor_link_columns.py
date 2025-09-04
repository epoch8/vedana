"""
Интеграционный тест: anchor_link_columns

Описание:
Reference / Reference List столбцы в таблицах Anchor_* должны порождать рёбра между
узлами согласно Data Model.

Данные:
- связь document <-> document_chunk задаётся колонкой в Anchor_document:
  link_document_has_document_chunk
- у документа 'document:1' в тестовых данных есть ссылки на чанки:
  document_chunk:01, document_chunk:02, document_chunk:03, document_chunk:05

Проверяем:
1) В DM находим линк document -> document_chunk и берем его sentence.
2) Получаем рёбра пайплайна (steps.get_grist_data()).
3) Убеждаемся, что в edges_df есть рёбра между 'document:1' и перечисленными чанками
   с нужным edge_label (из DM). Направление не важно.
"""

from typing import Set, Tuple

import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def _unordered(a: str, b: str) -> Tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def test_anchor_link_columns() -> None:
    # 1) Берём из Data Model sentence для document <-> document_chunk
    anchors_df, attrs_df, links_df = next(steps.get_data_model())
    assert not links_df.empty, "Data Model Links пуст."

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()
    row = dm[(a1 == "document") & (a2 == "document_chunk")]
    assert not row.empty, "В Data Model нет линка document -> document_chunk."
    sentence = str(row.iloc[0]["sentence"]).strip()
    assert sentence, "В Data Model для document <-> document_chunk пустой sentence."

    # 2) Получаем рёбра из пайплайна
    nodes_df, edges_df = next(steps.get_grist_data())
    assert isinstance(edges_df, pd.DataFrame) and not edges_df.empty, "edges_df пуст."

    # Фильтруем нужные рёбра: document <-> document_chunk, нужная метка
    ft = edges_df["from_node_type"].astype(str).str.lower().str.strip()
    tt = edges_df["to_node_type"].astype(str).str.lower().str.strip()
    lbl = edges_df["edge_label"].astype(str).str.lower().str.strip()

    target_edges = edges_df[
        (
            ((ft == "document") & (tt == "document_chunk"))
            | ((ft == "document_chunk") & (tt == "document"))
        )
        & (lbl == sentence.lower())
    ].copy()

    assert not target_edges.empty, (
        f"Не найдено ни одного ребра document <-> document_chunk с меткой '{sentence}'."
    )

    # Сформируем множество фактических неориентированных пар
    actual_pairs: Set[Tuple[str, str]] = set(
        _unordered(str(r["from_node_id"]).strip(), str(r["to_node_id"]).strip())
        for _, r in target_edges.iterrows()
    )

    # 3) Ожидаемые пары для 'document:1' из тестовых данных
    expected_pairs: Set[Tuple[str, str]] = {
        _unordered("document:1", "document_chunk:01"),
        _unordered("document:1", "document_chunk:02"),
        _unordered("document:1", "document_chunk:03"),
        _unordered("document:1", "document_chunk:05"),
    }

    missing = sorted(p for p in expected_pairs if p not in actual_pairs)
    assert not missing, (
        "Не все связи из reference-колонки Anchor_document попали в граф. "
        f"Отсутствуют пары: {missing}"
    )
