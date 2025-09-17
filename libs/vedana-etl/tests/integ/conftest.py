# libs/vedana-etl/tests/integ/conftest.py
import logging
import pytest

from dotenv import load_dotenv
from neo4j import GraphDatabase
from vedana_etl import steps
from vedana_core.settings import settings as core_settings

load_dotenv()


# -------- live Grist fixtures (NO mocks) --------
@pytest.fixture(scope="session")
def dm_dfs():
    """Data Model из живой Grist: Anchors, Attributes, Links."""
    anchors, attrs, links = next(steps.get_data_model())
    # sanity: типы как в коде
    assert attrs["embeddable"].dtype == bool
    assert "embed_threshold" in attrs.columns
    return anchors, attrs, links


@pytest.fixture(scope="session")
def raw_graph_dfs():
    """Сырые nodes/edges из живой Grist."""
    nodes, edges = next(steps.get_grist_data())
    return nodes, edges


# -------- optional live Memgraph ----------
def _ping_memgraph():
    try:
        drv = GraphDatabase.driver(
            core_settings.memgraph_uri, auth=(core_settings.memgraph_user, core_settings.memgraph_pwd)
        )
        with drv.session() as s:
            s.run("RETURN 1").consume()
        drv.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def live_memgraph_available():
    return _ping_memgraph()


# детерминированный провайдер эмбеддингов, чтобы тесты были воспроизводимы
@pytest.fixture
def dummy_llm(monkeypatch):
    class DummyProv:
        def create_embeddings_sync(self, texts):
            # фикс-вектор длины 8 (или сколько у тебя EMBEDDINGS_DIM — можно и динамически достать)
            return [[1.0] + [0.0] * (getattr(core_settings, "embeddings_dim", 8) - 1) for _ in texts]

    orig = steps.LLMProvider
    steps.LLMProvider = DummyProv  # type: ignore
    yield
    steps.LLMProvider = orig


@pytest.fixture(scope="session", autouse=True)
def quiet_logs():
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
