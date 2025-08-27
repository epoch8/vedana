"""
Интеграционный тест: anchor_attributes_formula_type_column

Цель:
  - Колонки Grist с типом данных "Formula" в сырых данных (get_grist_data)
    попадают как вычисленные значения (строки/числа и т.п., а не выражения).
  - Если такая колонка описана в Data Model, она должна сохраняться после filter_grist_nodes.

Test data:
  - Формульный атрибут: `document_filepath` (для узлов типа "document").
"""

from typing import Dict, Optional, Any

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_anchor_attributes_formula_type_column() -> None:
    """
    Проверяем поведение формульной колонки `document_filepath`:
    - в сырых nodes присутствует как результат вычисления (не пустое значение);
    - описана в Data Model и не отфильтровывается filter_grist_nodes.
    """

    # 1) Живой Data Model
    anchors_df, attrs_df, links_df = next(steps.get_data_model())
    assert not anchors_df.empty and not attrs_df.empty, "Data Model must not be empty."

    dm_attr_names = set(attrs_df["attribute_name"].astype(str))

    # В этом кейсе ожидаем, что формульный атрибут описан в Data Model.
    assert "document_filepath" in dm_attr_names, (
        "Test precondition failed: 'document_filepath' must be present in Data Model."
    )

    # 2) Данные из живой Grist
    nodes_df, _ = next(steps.get_grist_data(batch_size=1000))
    assert not nodes_df.empty, "No nodes fetched from Grist."

    documents = nodes_df[nodes_df["node_type"] == "document"]
    assert not documents.empty, "Expected at least one 'document' node in raw data."

    # 3) Найдём хотя бы одно непустое значение document_filepath в raw
    raw_value: Optional[Any] = None
    raw_node_id: Optional[str] = None
    for _, row in documents.iterrows():
        attrs: Dict[str, Any] = row["attributes"] or {}
        if "document_filepath" in attrs and str(attrs["document_filepath"]).strip():
            raw_value = attrs["document_filepath"]
            raw_node_id = row["node_id"]
            break

    assert raw_value is not None and str(raw_value).strip(), (
        """
        Formula column 'document_filepath' must appear as a computed, non-empty value 
        in at least one 'document' node (raw data).
        """
    )

    # 4) После фильтрации по Data Model атрибут должен сохраниться
    filtered = steps.filter_grist_nodes(nodes_df, dm_nodes=anchors_df, dm_attributes=attrs_df)
    docs_f = filtered[filtered["node_type"] == "document"]
    assert not docs_f.empty, "Filtered graph should still contain 'document' nodes."

    # Проверим, что у того же узла (если он остался) поле всё ещё есть и непустое
    if raw_node_id is not None and (docs_f["node_id"] == raw_node_id).any():
        row = docs_f.loc[docs_f["node_id"] == raw_node_id].iloc[0]
        attrs_f = row["attributes"] or {}
        assert "document_filepath" in attrs_f and str(attrs_f["document_filepath"]).strip(), (
            "Expected 'document_filepath' to be preserved by filter_grist_nodes because it is present in Data Model."
        )
    else:
        # Иначе просто убедимся, что у какого-то document-узла поле присутствует
        found_any = False
        for _, row in docs_f.iterrows():
            attrs_f: Dict[str, Any] = row["attributes"] or {}
            if "document_filepath" in attrs_f and str(attrs_f["document_filepath"]).strip():
                found_any = True
                break
        assert found_any, (
            """
            Expected 'document_filepath' to be present on at least one 'document' node 
            after filter_grist_nodes (it's in Data Model).
            """
        )
