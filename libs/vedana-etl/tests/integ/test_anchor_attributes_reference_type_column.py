"""
Интеграционный тест: anchor_attributes_reference_type_column (SQL Reference)

Проверяем логику восстановления Reference-полей при чтении из Grist SQL:
 - SQL отдаёт <ref_id> и дополнительный "gristHelper_<col>" со строковым значением.
 - Провайдер должен подставить именно строковое значение в attributes и НЕ протаскивать
   вспомогательные gristHelper_* ключи в графовые атрибуты.

Требования:
 - Если Reference-колонка есть в Data Model -> должна остаться после filter_grist_nodes.
 - Колонки, которых нет в Data Model (например, document_random_attr), должны исчезнуть.

Тестовые данные:
 - reference-поле: document_reference_attr (есть в DM)
 - лишнее поле:    document_random_attr (нет в DM)
"""

from typing import Dict, Optional, Any

from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_anchor_attributes_reference_type_column() -> None:
    """
    1) В сырых nodes (get_grist_data) у хотя бы одного document-узла
       `document_reference_attr` присутствует как непустая строка.
       В attributes не должно быть gristHelper_* ключей.
    2) После filter_grist_nodes:
       - `document_reference_attr` остаётся (т.к. есть в DM),
       - `document_random_attr` исчезает (т.к. нет в DM).
    """

    # --- 1) Живой Data Model
    anchors_df, attrs_df, _ = next(steps.get_data_model())
    assert not anchors_df.empty and not attrs_df.empty, "Data Model must not be empty."

    dm_attr_names = set(attrs_df["attribute_name"].astype(str))
    assert "document_reference_attr" in dm_attr_names, (
        "Precondition: 'document_reference_attr' must be present in Data Model."
    )
    assert "document_random_attr" not in dm_attr_names, (
        "Precondition: 'document_random_attr' must be absent from Data Model."
    )

    # --- 2) Данные из живой Grist
    nodes_df, _ = next(steps.get_grist_data(batch_size=1000))
    assert not nodes_df.empty, "No nodes fetched from Grist."

    documents = nodes_df[nodes_df["node_type"] == "document"]
    assert not documents.empty, "Expected at least one 'document' node in raw data."

    # --- 2.1) Найти document со строковым document_reference_attr
    ref_node_attrs: Optional[Dict[str, Any]] = None
    for _, row in documents.iterrows():
        attrs: Dict[str, Any] = row["attributes"] or {}
        val = attrs.get("document_reference_attr")
        if isinstance(val, str) and val.strip():
            ref_node_attrs = attrs
            break

    assert ref_node_attrs is not None, (
        """
        Reference column 'document_reference_attr' must appear as a non-empty string 
        in at least one 'document' node (raw data).
        """
    )

    # --- 2.2) Проверить, что gristHelper_* ключи не протекли в attributes
    assert not any(k.startswith("gristHelper_") for k in ref_node_attrs.keys()), (
        """
        gristHelper_* keys leaked into attributes; the SQL provider should have 
        used them to reconstruct the final string value and NOT keep helper keys.
        """
    )

    # --- 3) Фильтрация по Data Model
    filtered = steps.filter_grist_nodes(nodes_df, dm_nodes=anchors_df, dm_attributes=attrs_df)
    docs_f = filtered[filtered["node_type"] == "document"]
    assert not docs_f.empty, "Filtered graph should still contain 'document' nodes."

    # 3.1) document_reference_attr должен сохраниться (он в Data Model)
    preserved_any = False
    for _, row in docs_f.iterrows():
        attrs: Dict[str, Any] = row["attributes"] or {}
        if isinstance(attrs.get("document_reference_attr"), str) and attrs["document_reference_attr"].strip():
            preserved_any = True
            break
    assert preserved_any, (
        "Expected 'document_reference_attr' to be preserved by filter_grist_nodes because it is present in Data Model."
    )

    # 3.2) document_random_attr должен быть удалён (его нет в Data Model)
    offenders = [
        row["node_id"]
        for _, row in docs_f.iterrows()
        if "document_random_attr" in (row["attributes"] or {})
    ]
    assert not offenders, (
        f"""
        'document_random_attr' unexpectedly survived filter_grist_nodes, 
        but it's not in Data Model. Offenders: {offenders}
        """
    )
