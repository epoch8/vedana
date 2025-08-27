"""
Интеграционный тест: anchor_embeddable_attributes

Проверяем правило:
  - эмбеддятся только атрибуты с embeddable == True
  - и тип данных НЕ входит в {bool, float, int}.
    (пустой dtype трактуем как допустимый строкоподобный)

Шаги:
  1) Загружаем Data Model из живой Grist: steps.get_data_model().
  2) Считаем ожидаемый набор имён атрибутов по правилу из описания.
  3) Подключаемся к живому Memgraph (bolt://localhost:7687 по умолчанию),
     вызываем steps.ensure_memgraph_indexes(attrs_df) — индексы реально создаются.
  4) Сравниваем фактический набор (memgraph_vector_indexes.attribute_name) с ожидаемым:
     - векторизуемые НЕ должны содержать dtype IN {bool,float,int}
     - должны включать ключевые строковые embeddable атрибуты (например, document_name, document_chunk_text)
     - фактический набор допустимо быть подмножеством «ожидаемого с пустыми dtype»,
       т.к. приложение может ужесточать правило до dtype == "str".
"""

from typing import Set

from dotenv import load_dotenv
import pandas as pd

from vedana_etl import steps

load_dotenv()


def test_anchor_embeddable_attributes() -> None:
    """
    Инварианты:
      - memgraph_vector_indexes содержит только embeddable атрибуты «строкового семейства»
        (исключая bool/float/int; пустой dtype допустим как строкоподобный).
      - набор включает ключевые текстовые поля из тестовой Data Model.
    """

    # 1) Живой data model
    anchors_df, attrs_df, links_df = next(steps.get_data_model())

    # 2) Ожидаемый набор по правилу (NOT IN {bool,float,int}; empty -> допустим)
    dtype_norm: pd.Series = (
        attrs_df["dtype"]
        .where(attrs_df["dtype"].notna(), "")
        .astype(str)
        .str.strip()
        .str.lower()
    )
    embeddable_mask: pd.Series = attrs_df["embeddable"].astype(bool)

    forbidden: Set[str] = {"bool", "float", "int"}
    allowed_type_mask: pd.Series = ~dtype_norm.isin(forbidden)

    expected_with_empty_ok: Set[str] = set(
        attrs_df.loc[embeddable_mask & allowed_type_mask, "attribute_name"].astype(str)
    )

    # sanity: в данных должны присутствовать запрещённые типы (чтобы проверка что-то значила)
    assert any(t in forbidden for t in dtype_norm.unique()), (
        "Test DM should contain at least one non-string dtype (bool/float/int)."
    )
    assert expected_with_empty_ok, "No embeddable string-like attributes found in DM."

    # 3) Реально создаём индексы в Memgraph и берём фактический набор
    _, mem_vec_idx = steps.ensure_memgraph_indexes(attrs_df)
    actual: Set[str] = set(mem_vec_idx["attribute_name"].astype(str))

    # 4) Проверки

    # 4.1. фактический набор не должен включать запрещённые по типу атрибуты
    forbidden_in_actual = set(
        attrs_df.loc[
            attrs_df["attribute_name"].astype(str).isin(actual) & dtype_norm.isin(forbidden),
            "attribute_name",
        ].astype(str)
    )
    assert not forbidden_in_actual, f"Forbidden dtypes leaked into vectorizable set: {sorted(forbidden_in_actual)}"

    # 4.2. фактический набор должен быть подмножеством ожидаемого «с пустыми как допустимыми»
    # (текущая реализация может требовать строго dtype == 'str', это частный случай правила из описания)
    assert actual.issubset(expected_with_empty_ok), (
        "Vectorizable attributes include names outside allowed rule (embeddable & non-numeric/bool). "
        f"Unexpected extras: {sorted(actual - expected_with_empty_ok)}"
    )

    # 4.3. must-have: убедимся, что ключевые текстовые embeddable точно попали
    # (подгони список под твою тестовую DM при необходимости)
    must_have = {"document_name", "document_chunk_text"}
    missing = sorted(must_have - actual)
    assert not missing, f"Expected vectorizable attributes missing: {missing}"
