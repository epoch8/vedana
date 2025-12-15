"""
Интеграционный тест: table_filtering

Цель:
  В пайплайн должны попадать только таблицы с префиксами Anchor_/Link_.
  Практически это означает:
    - В nodes попадают только типы узлов, перечисленные в Data Model (Anchors).
    - Любые прочие таблицы (например, meta_document_reference_attrs) не должны попадать в пайплайн.

Данные:
  На тестовой Grist есть служебная таблица `meta_document_reference_attrs`, она НЕ должна загружаться в пайплайн.

Проверки:
  1) Множество уникальных node_type из get_grist_data() (кроме служебного 'DataModel')
  является подмножеством {noun из Anchors}.
  2) В node_type нет значений, начинающихся с 'meta_' (case-insensitive).
"""

from typing import Set

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_table_filtering() -> None:
    """
    Проверяем, что в граф попадают только таблицы Anchor_/Link_, а meta_* не грузится.
    """

    # 1) Загружаем Data Model (Anchors/Attributes/Links) и сырые узлы/рёбра из живой Grist.
    anchors_df, a_attrs_df, l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())
    nodes_df, edges_df = next(steps.get_grist_data())

    # Проверяем наичие данных в Grist.
    assert not anchors_df.empty, "Anchors in Data Model must not be empty."
    assert not nodes_df.empty, "No nodes fetched from Grist."

    # 2) Разрешённые типы узлов — только те, что перечислены в Anchors нотации Data Model.
    allowed_node_types: Set[str] = set(anchors_df["noun"].astype(str))

    # 'DataModel' всегда присутствует — исключаем его из проверки фильтрации.
    actual_node_types: Set[str] = set(nodes_df.loc[nodes_df["node_type"] != "DataModel", "node_type"].astype(str))

    # 2.1) Все типы узлов из данных должны соответствовать якорям Data Model.
    assert actual_node_types.issubset(allowed_node_types), f"""
        Found node types that do not correspond to Data Model Anchors (allowed={sorted(allowed_node_types)}, 
        actual={sorted(actual_node_types)}, extra={sorted(actual_node_types - allowed_node_types)})
        """

    # 3) Дополнительно убеждаемся, что meta-таблица не превратилась в узлы.
    # Проверяем, что в типах нет ничего начинающегося с 'meta_' (регистр игнорируем).
    lower_types = {t.lower() for t in actual_node_types}
    banned_prefix = "meta_"
    offending = sorted(t for t in lower_types if t.startswith(banned_prefix))
    assert not offending, f"Meta-table leaked into nodes: {offending}"

    # Фиксируем наличие хотя бы одного «нормального» типа из Data Model.
    assert any(t in lower_types for t in ("document", "document_chunk", "regulation")), """
        Expected at least one of typical anchor types ('document', 'document_chunk', 'regulation') 
        to be present in the nodes. Adjust this assertion if your DM differs.
        """
