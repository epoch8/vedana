"""
Интеграционный тест: anchor_attributes_reference_type_column (SQL Reference)

Фокус теста:
  - При чтении через Grist SQL провайдер Reference-колонка приходит как <ref_id> + gristHelper_<col>,
    а в итоговом attributes должно быть **строковое значение** (как в UI), без gristHelper_* ключей.
  - Если Reference-колонка описана в Data Model, она должна сохраниться после filter_grist_nodes.

Тестовые данные:
 - reference-поле: document_reference_attr (есть в Data Model)
"""

from typing import Any, Dict, Optional

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_anchor_attributes_reference_type_column() -> None:
    """
    1) В сыром nodes (get_grist_data) у хотя бы одного document-узла
       `document_reference_attr` присутствует как непустая строка.
       В attributes не должно быть gristHelper_* ключей.
    2) После filter_grist_nodes:
       - `document_reference_attr` остаётся (т.к. она есть в Data Model).
    """

    # --- 1) Живой Data Model
    anchors_df, attrs_df, _ = next(steps.get_data_model())
    assert not anchors_df.empty and not attrs_df.empty, "Data Model must not be empty."

    dm_attr_names = set(attrs_df["attribute_name"].astype(str))
    assert "document_reference_attr" in dm_attr_names, (
        "Precondition: 'document_reference_attr' must be present in Data Model."
    )

    # --- 2) Данные из живой Grist
    nodes_df, _ = next(steps.get_grist_data())
    assert not nodes_df.empty, "No nodes fetched from Grist."

    documents = nodes_df[nodes_df["node_type"] == "document"]
    assert not documents.empty, "Expected at least one 'document' node in raw data."

    # --- 2.1) Найти document со строковым document_reference_attr
    ref_node_attrs: Optional[Dict[str, Any]] = None
    for _, row in documents.iterrows():
        attrs: Dict[str, Any] = row["attributes"] or {}
        print(attrs)
        val = attrs.get("document_reference_attr")
        if isinstance(val, int):
            ref_node_attrs = attrs
            break

    assert ref_node_attrs is not None, """
        Reference column 'document_reference_attr' must appear as a non-empty string 
        in at least one 'document' node (raw data).
        """

    # --- 2.2) Проверить, что gristHelper_* ключи не протекли в attributes
    assert not any(k.startswith("gristHelper_") for k in ref_node_attrs.keys()), """
        gristHelper_* keys leaked into attributes; the SQL provider should have 
        used them to reconstruct the final string value and NOT keep helper keys.
        """

    # --- 3) Фильтрация по Data Model
    filtered = steps.filter_grist_nodes(nodes_df, dm_nodes=anchors_df, dm_attributes=attrs_df)
    docs_f = filtered[filtered["node_type"] == "document"]
    assert not docs_f.empty, "Filtered graph should still contain 'document' nodes."

    preserved_any = False
    for _, row in docs_f.iterrows():
        attrs = row["attributes"] or {}
        if isinstance(attrs.get("document_reference_attr"), int):
            preserved_any = True
            break
    assert preserved_any, (
        "Expected 'document_reference_attr' to be preserved by filter_grist_nodes because it is present in Data Model."
    )
