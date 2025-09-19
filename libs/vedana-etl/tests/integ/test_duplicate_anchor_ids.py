"""
Интеграционный тест: duplicate_anchor_ids

Описание:
  При дублировании ID узлов пайплайн не должен ломаться.
  В граф должен попасть первый описанный узел для данного node_id.

Тестовые данные:
  В Anchor_document есть два объекта с одним и тем же node_id "document:1".
  Ожидается, что после drop_duplicates по "node_id" останется первая запись
  (в снапшоте тестового Grist это документ с document_name = "doc_a").

Проверки:
  1) В финальном nodes_df присутствует ровно один узел с node_id == "document:1".
  2) Его атрибуты соответствуют первой записи (document_name == "doc_a").
"""

from typing import Dict

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_duplicate_anchor_ids_keep_first() -> None:
    """
    Убеждаемся, что при наличии дублей по node_id остаётся первая запись.
    """

    # 1) Грузим графовые таблицы из живой Grist через стандартный шаг пайплайна.
    nodes_df, _ = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes fetched from Grist; check test data and GRIST_* env."

    # 2) Фильтруем по проблемному идентификатору тестовых данных.
    masked = nodes_df[nodes_df["node_id"] == "document:1"]
    assert not masked.empty, "Expected at least one node with node_id == 'document:1' in test data."

    # 3) Дубликаты должны быть схлопнуты: остаётся ровно одна строка.
    assert len(masked) == 1, f"Duplicate node_id 'document: 1' wasn't deduplicated. Found {len(masked)} rows."

    # 4) Проверяем, что осталась именно первая запись (ожидаемое имя 'doc_a').
    attrs: Dict[str, object] = masked.iloc[0]["attributes"] or {}
    got_name = attrs.get("document_name")
    assert got_name == "doc_a", f"""
        Deduplication didn't preserve the first record for 'document:1'. 
        Expected document_name == 'doc_a', got {got_name!r}.
        """
