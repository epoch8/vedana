# libs/vedana-etl/tests/integ/conftest.py
import logging

import pytest
from dotenv import load_dotenv
from neo4j import GraphDatabase
from vedana_core.settings import settings as core_settings

from vedana_etl import steps

load_dotenv()


# -------- live Grist fixtures (NO mocks) --------
@pytest.fixture(scope="session")
def dm_dfs():
    """Data Model from live Grist: Anchors, Attributes, Links."""
    anchors_df, a_attrs_df, l_attrs_df, links_df, q_df, p_df, cl_df = next(steps.get_data_model())
    # sanity: types as in code
    assert a_attrs_df["embeddable"].dtype == bool and l_attrs_df["embeddable"].dtype == bool
    assert "embed_threshold" in a_attrs_df.columns and "embed_threshold" in l_attrs_df.columns
    return anchors_df, a_attrs_df, l_attrs_df, links_df, q_df, p_df, cl_df


@pytest.fixture(scope="session")
def raw_graph_dfs():
    """Raw nodes/edges from live Grist."""
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


# deterministic embeddings provider for reproducible tests
@pytest.fixture
def dummy_llm(monkeypatch):
    class DummyProv:
        def create_embeddings_sync(self, texts):
            # fixed vector of length 8 (or however many EMBEDDINGS_DIM you have - can be fetched dynamically)
            return [[1.0] + [0.0] * (getattr(core_settings, "embeddings_dim", 8) - 1) for _ in texts]

    orig = steps.LLMProvider
    steps.LLMProvider = DummyProv  # type: ignore
    yield
    steps.LLMProvider = orig


@pytest.fixture(scope="session", autouse=True)
def quiet_logs():
    logging.getLogger("neo4j").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.ERROR)
