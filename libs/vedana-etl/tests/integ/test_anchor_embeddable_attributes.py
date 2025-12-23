"""
Интеграционный тест: anchor_embeddable_attributes

Правило:
  эмбеддятся только атрибуты с embeddable == True и dtype ∈ {"str", ""} (пустой — как строкоподобный)

Шаги:
  1) Загружаем Data Model из живой Grist.
  2) Строим ожидаемый набор имён атрибутов по правилу выше.
  3) Запускаем ensure_memgraph_indexes(...) на живом Memgraph, получаем фактический набор.
  4) Проверяем: фактический набор — подмножество ожидаемого; и не содержит атрибутов с «запрещённым» dtype.
"""

from typing import Set

from dotenv import load_dotenv
import pandas as pd

from vedana_etl import steps

load_dotenv()


def test_anchor_embeddable_attributes() -> None:
    """
    Инварианты:
      - memgraph_vector_indexes содержит только embeddable атрибуты с dtype ∈ {"str", ""}.
      - в наборе есть ключевые текстовые поля (document_name, document_chunk_text).
    """

    # 1) Живой data model
    anchors_df, a_attrs_df, _l_attrs_df, links_df, _q_df, _p_df, _cl_df = next(steps.get_data_model())

    # 2) Ожидаемый набор по правилу (IN {str}; empty -> допустим)
    dtype_norm: pd.Series = a_attrs_df["dtype"].where(a_attrs_df["dtype"].notna(), "").astype(str).str.strip().str.lower()
    embeddable_mask: pd.Series = a_attrs_df["embeddable"].astype(bool)

    # Белый список типов, при желании можно расширить: {"str", "", "string", "text"}
    allowed_str_like = {"str", ""}
    allowed_type_mask: pd.Series = dtype_norm.isin(allowed_str_like)

    expected_allowed: Set[str] = set(a_attrs_df.loc[embeddable_mask & allowed_type_mask, "attribute_name"].astype(str))

    assert expected_allowed, "No embeddable attributes with dtype in {'str',''} found in Data Model."

    # 3) Реально создаём индексы в Memgraph и берём фактический набор
    _, mem_vec_idx = steps.ensure_memgraph_node_indexes(a_attrs_df)
    actual: Set[str] = set(mem_vec_idx["attribute_name"].astype(str))

    # 4.1. Фактический набор — подмножество ожидаемого белого списка
    extras = sorted(actual - expected_allowed)
    assert actual.issubset(expected_allowed), f"""
        Vectorizable attributes include names outside the allowed rule (embeddable & dtype in {"str", ""}). 
        "Unexpected extras: {extras}
        """

    # 4.2. must-have: ключевые текстовые поля точно присутствуют
    must_have = {"document_name", "document_chunk_text"}
    missing = sorted(must_have - actual)
    assert not missing, f"Expected vectorizable attributes missing: {missing}"
