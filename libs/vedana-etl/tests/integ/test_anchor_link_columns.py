"""
Интеграционный тест: anchor_link_columns

Описание:
Reference / Reference List столбцы в таблицах Anchor_* должны порождать рёбра
между узлами, как это описано в Data Model.

Данные:
Связь document <-> document_chunk.
В Data Model:
- anchor1 = document, anchor2 = document_chunk
- anchor1_link_column_name = link_document_has_document_chunk  (в Anchor_document)
- anchor2_link_column_name = link_document_chunk_document_id   (в Anchor_document_chunk)  # если задано

Проверяем:
1) В DM есть линк document->document_chunk; берём его sentence как метку ребра.
2) По обеим link-колонкам собираем ожидаемые пары (document_id, document_chunk_id):
   - поддерживаем типы: list / int / str, включая строку с перечислением через запятую.
3) В edges_df выбираем рёбра document <-> document_chunk с edge_label == sentence.
4) Фактическое множество пар должно покрывать ожидаемое.
5) Для конкретного 'document:1' проверяем, что из его колонки действительно получились связи.
"""

from typing import Any, Dict, List, Set, Tuple

import math
import pandas as pd
import pytest
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def _split_comma_list(s: str) -> List[str]:
    """
    Разбить строку по запятым в список строк-ID, убрать пустые / пробелы.
    """

    return [part.strip() for part in s.split(",") if part.strip()]


def _listify_refs(val: Any) -> List[str]:
    """
    Нормализовываем значение Reference / Reference List к списку строковых id:
    - list -> как есть (str/int -> str)
    - int -> ["<int>"]
    - str -> либо ["str"], либо парсим comma-separated "a, b, c"
    - None/NaN -> []
    """

    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    if isinstance(val, int):
        return [str(val)]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        # если в строке несколько ссылок — поддержим CSV-формат
        return _split_comma_list(s) if "," in s else [s]
    # любой другой тип приводим к строке
    s = str(val).strip()
    return [s] if s else []


def _unordered_pair(a: str, b: str) -> Tuple[str, str]:
    """Неориентированная пара (для сравнения множеств пар ребёр)."""
    return (a, b) if a <= b else (b, a)


def test_anchor_link_columns() -> None:
    # 1) Data Model: найдём линк document -> document_chunk и имена link-колонок
    anchors_df, attrs_df, links_df = next(steps.get_data_model())
    assert not links_df.empty, "DM Links пуст — проверь настройки GRIST_*."

    dm = links_df.copy()
    a1 = dm["anchor1"].astype(str).str.lower().str.strip()
    a2 = dm["anchor2"].astype(str).str.lower().str.strip()
    row = dm[(a1 == "document") & (a2 == "document_chunk")]
    assert not row.empty, "В DM нет линка document -> document_chunk."

    row = row.iloc[0]
    sentence = str(row["sentence"]).strip()
    col_from = str(row.get("anchor1_link_column_name") or "").strip()  # в Anchor_document
    col_to = str(row.get("anchor2_link_column_name") or "").strip()    # в Anchor_document_chunk (если есть)
    assert sentence, "В DM для ссылки задан пустой sentence."
    assert col_from or col_to, "В DM не задано ни одной link-колонки для document<->document_chunk."

    # 2) Данные из Grist (узлы + рёбра)
    nodes_df, edges_df = next(steps.get_grist_data())
    assert not nodes_df.empty, "Нет узлов из Grist."
    assert isinstance(edges_df, pd.DataFrame), "edges_df должен быть DataFrame."

    docs = nodes_df[nodes_df["node_type"].astype(str).str.lower() == "document"].copy()
    chunks = nodes_df[nodes_df["node_type"].astype(str).str.lower() == "document_chunk"].copy()
    assert not docs.empty and not chunks.empty, "В данных отсутствуют document или document_chunk."

    # Множество существующих node_id (на всякий — иногда полезно валидировать пары)
    doc_ids: Set[str] = set(docs["node_id"].astype(str))
    chunk_ids: Set[str] = set(chunks["node_id"].astype(str))

    # 2.1) Ожидаемые пары из стороны document (anchor1)
    expected_pairs: Set[Tuple[str, str]] = set()
    if col_from:
        for _, r in docs.iterrows():
            doc_id = str(r["node_id"])
            attrs: Dict[str, Any] = r.get("attributes") or {}
            for chunk_id in _listify_refs(attrs.get(col_from)):
                # Добавляем только если chunk реально существует (защита от мусора)
                if chunk_id in chunk_ids:
                    expected_pairs.add(_unordered_pair(doc_id, chunk_id))

    # 2.2) Ожидаемые пары из стороны chunk (anchor2), если такая колонка задана
    if col_to:
        for _, r in chunks.iterrows():
            chunk_id = str(r["node_id"])
            attrs: Dict[str, Any] = r.get("attributes") or {}
            for doc_id in _listify_refs(attrs.get(col_to)):
                if doc_id in doc_ids:
                    expected_pairs.add(_unordered_pair(doc_id, chunk_id))

    assert expected_pairs, (
        f"Не удалось собрать ни одной ожидаемой пары по колонкам "
        f"{col_from or '-'} / {col_to or '-'}. Проверьте тестовые данные."
    )

    # 3) Фактические рёбра document <-> document_chunk с нужной меткой
    if edges_df.empty:
        pytest.fail("edges_df пуст.")

    ft = edges_df["from_node_type"].astype(str).str.lower().str.strip()
    tt = edges_df["to_node_type"].astype(str).str.lower().str.strip()
    lbl = edges_df["edge_label"].astype(str).str.lower().str.strip()

    er = edges_df[
        (
            ((ft == "document") & (tt == "document_chunk")) |
            ((ft == "document_chunk") & (tt == "document"))
        )
        & (lbl == sentence.lower())
    ].copy()

    assert not er.empty, "Не найдено ни одного ребра document <-> document_chunk с меткой из DM."

    # 3.1) Построим множество фактических пар (неориентированных)
    actual_pairs: Set[Tuple[str, str]] = set(
        _unordered_pair(str(r["from_node_id"]), str(r["to_node_id"])) for _, r in er.iterrows()
    )

    # 4) Проверки покрытия
    missing = sorted(pair for pair in expected_pairs if pair not in actual_pairs)
    assert not missing, (
        "Не все связи из Reference-колонок построены в графе. "
        f"Отсутствуют пары: {missing}"
    )

    # 5) Точная проверка для 'document:1' (если он есть): пары из его колонки должны появиться
    doc1 = docs[docs["node_id"].astype(str) == "document:1"]
    if not doc1.empty and col_from:
        doc1_attrs: Dict[str, Any] = doc1.iloc[0].get("attributes") or {}
        expected_for_doc1 = set(
            _unordered_pair("document:1", cid) for cid in _listify_refs(doc1_attrs.get(col_from)) if cid in chunk_ids
        )
        if expected_for_doc1:
            missing_for_doc1 = sorted(pair for pair in expected_for_doc1 if pair not in actual_pairs)
            assert not missing_for_doc1, (
                f"Для 'document: 1' отсутствуют связи по колонке '{col_from}': {missing_for_doc1}"
            )
