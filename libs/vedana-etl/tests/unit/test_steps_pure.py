import pandas as pd
from dotenv import load_dotenv

from vedana_etl import steps

load_dotenv()


def test_clean_str_replaces_and_collapses_spaces():
    s = "A\u00a0B\u2009C\u200b  D\tE"
    # NBSP, thin space, zero-width + мультипробелы -> одиночные пробелы
    assert steps.clean_str(s) == "A B C D E"


def test_clean_str_passthrough_non_str():
    assert steps.clean_str("123") == "123"


def test_is_uuid_true_false():
    assert steps.is_uuid("550e8400-e29b-41d4-a716-446655440000")
    assert not steps.is_uuid("550e8400e29b41d4a716446655440000X")
    assert not steps.is_uuid("not-a-uuid")


def test_generate_embeddings_for_nodes(monkeypatch):
    df = pd.DataFrame(
        [
            {"node_id": "a1", "node_type": "Article", "attributes": {"title": "hello", "year": 2020}},
            {"node_id": "u1", "node_type": "Author", "attributes": {"name": "Bob"}},  # без векторизации
        ]
    )

    dm_node_attrs = pd.DataFrame(
        [
            {
                "attribute_name": "title",
                "anchor": "Article",
                "embeddable": True,
                "dtype": "str",
                "embed_threshold": 0.8,
            }
        ]
    )

    class DummyProv:
        def create_embeddings_sync(self, texts):
            # ожидаем только один текст 'hello'
            assert texts == ["hello"]
            return [[1.0, 0.0]]

    orig = steps.LLMProvider
    try:
        steps.LLMProvider = DummyProv  # type: ignore
        out = steps.generate_embeddings(df.copy(), dm_node_attrs)
    finally:
        steps.LLMProvider = orig

    attrs = out[out["node_id"] == "a1"].iloc[0]["attributes"]
    assert attrs["title_embedding"] == [1.0, 0.0]
    # у автора нет добавленного embedding
    assert "title_embedding" not in out[out["node_id"] == "u1"].iloc[0]["attributes"]


def test_generate_embeddings_skips_uuid_text(monkeypatch):
    uuid_text = "550e8400-e29b-41d4-a716-446655440000"
    df = pd.DataFrame(
        [
            {"node_id": "a1", "node_type": "Article", "attributes": {"title": uuid_text}},
        ]
    )

    dm_node_attrs = pd.DataFrame(
        [
            {
                "attribute_name": "title",
                "anchor": "Article",
                "embeddable": True,
                "dtype": "str",
                "embed_threshold": 0.8,
            }
        ]
    )

    class DummyProv:
        def create_embeddings_sync(self, texts):
            # не должен вызываться, т.к. текст выглядит как UUID
            raise AssertionError("LLM should not be called for UUID-like text")

    orig = steps.LLMProvider
    try:
        steps.LLMProvider = DummyProv  # type: ignore
        out = steps.generate_embeddings(df.copy(), dm_node_attrs)
    finally:
        steps.LLMProvider = orig

    assert out.iloc[0]["attributes"] == {"title": uuid_text}


def test_generate_embeddings_for_edges(monkeypatch):
    df = pd.DataFrame(
        [
            {
                "from_node_id": "u1",
                "to_node_id": "a1",
                "from_node_type": "Author",
                "to_node_type": "Article",
                "edge_label": "WROTE",
                "attributes": {"title": "edge text"},
            },
        ]
    )

    dm_link_attrs = pd.DataFrame(
        [
            {
                "attribute_name": "title",
                "link": "WROTE",
                "embeddable": True,
                "dtype": "str",
                "embed_threshold": 0.8,
            }
        ]
    )

    class DummyProv:
        def create_embeddings_sync(self, texts):
            assert texts == ["edge text"]
            return [[0.5, 0.5]]

    orig = steps.LLMProvider
    try:
        steps.LLMProvider = DummyProv  # type: ignore
        out = steps.generate_embeddings(df.copy(), dm_link_attrs)
    finally:
        steps.LLMProvider = orig

    attrs = out.iloc[0]["attributes"]
    assert attrs["title_embedding"] == [0.5, 0.5]
