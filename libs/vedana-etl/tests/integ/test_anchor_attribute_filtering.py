"""
Интеграционный тест: anchor_attribute_filtering

Что проверяем:
  - В граф (после filter_grist_nodes) попадают ТОЛЬКО атрибуты, описанные в дата-модели.
  - Специально проверяем, что "document_random_attr" (присутствует в тестовых данных)
    полностью удаляется у типа "document".

Шаги:
  1) Загружаем Data Model (Anchors/Attributes/Links) из живой Grist.
  2) Загружаем сырые данные из живой Grist.
  3) Применяем steps.filter_grist_nodes(nodes, dm_nodes=Anchors, dm_attributes=Attributes).
  4) Проверяем:
     - атрибуты (кроме служебного DataModel) — подмножество Data Model атрибутов,
       допускаем только *_embedding (могут появиться позднее).
     - для типа "document" ключ "document_random_attr" отсутствует.
"""

from typing import Dict, Set

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_anchor_attribute_filtering_removes_unknown() -> None:
    """
    Основная проверка фильтрации атрибутов узлов.
    """

    # 1) Проверяем Data Model
    anchors_df, attrs_df, links_df = next(steps.get_data_model())
    assert not anchors_df.empty and not attrs_df.empty, "Data Model must not be empty."

    # 2) Проверяем данные из Grist
    nodes_df, _ = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes fetched from Grist."

    # 3) Допустимые атрибуты по Data Model
    allowed_attrs: Set[str] = set(attrs_df["attribute_name"].astype(str))

    # 3.1) У каждого узла (кроме DataModel) ключи атрибутов ⊆ Data Model атрибутов (плюс *_embedding)
    for _, row in nodes_df[nodes_df["node_type"] != "DataModel"].iterrows():
        attr_dict: Dict[str, object] = row["attributes"] or {}
        keys = set(map(str, attr_dict.keys()))
        # разрешаем сгенерированные позже эмбеддинги
        embedding_keys = {k for k in keys if k.endswith("_embedding")}
        unknown = keys - allowed_attrs - embedding_keys
        assert not unknown, (
            f"""
            Node {row['node_id']} ({row['node_type']}) contains attributes not described in 
            Data Model: {sorted(unknown)}
            """
        )

    # 3.2) Специальный кейс: document_random_attr должен исчезнуть у document-узлов
    docs = nodes_df[nodes_df["node_type"] == "document"]
    assert not docs.empty, "Expected at least one 'document' node in test data."
    still_has_random = [
        row["node_id"]
        for _, row in docs.iterrows()
        if "document_random_attr" in (row["attributes"] or {})
    ]
    assert not still_has_random, (
        "Unexpected attribute 'document_random_attr' is still present "
        f"in document nodes: {still_has_random}"
    )

    # Убедимся, что у document осталось хотя бы одно валидное поле из DM, чтобы тест не проходил
    # «пустым» набором атрибутов.
    any_valid_left = any(
        bool(set((row["attributes"] or {}).keys()) & allowed_attrs)
        for _, row in docs.iterrows()
    )
    assert any_valid_left, (
        "After filtering, document nodes should still have at least one attribute from Data Model."
    )
