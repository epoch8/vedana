import pytest
import pytest_asyncio
import sqlalchemy as sa
import sqlalchemy.ext.asyncio as sa_aio
from sqlalchemy.pool import StaticPool

from vedana_core.data_model import DataModel

CREATE_REQUIRED_TABLES_SQL = [
    """
    CREATE TABLE dm_anchors (
        noun TEXT PRIMARY KEY,
        description TEXT,
        id_example TEXT,
        query TEXT
    )
    """,
    """
    CREATE TABLE dm_links (
        anchor1 TEXT NOT NULL,
        anchor2 TEXT NOT NULL,
        sentence TEXT NOT NULL,
        description TEXT,
        query TEXT,
        anchor1_link_column_name TEXT,
        anchor2_link_column_name TEXT,
        has_direction BOOLEAN,
        PRIMARY KEY (anchor1, anchor2, sentence)
    )
    """,
    """
    CREATE TABLE dm_anchor_attributes (
        anchor TEXT NOT NULL,
        attribute_name TEXT NOT NULL,
        description TEXT,
        data_example TEXT,
        embeddable BOOLEAN,
        query TEXT,
        dtype TEXT,
        embed_threshold REAL,
        PRIMARY KEY (anchor, attribute_name)
    )
    """,
    """
    CREATE TABLE dm_link_attributes (
        link TEXT NOT NULL,
        attribute_name TEXT NOT NULL,
        description TEXT,
        data_example TEXT,
        embeddable BOOLEAN,
        query TEXT,
        dtype TEXT,
        embed_threshold REAL,
        PRIMARY KEY (link, attribute_name)
    )
    """,
]

CREATE_OPTIONAL_TABLES_SQL = [
    """
    CREATE TABLE dm_queries (
        query_name TEXT PRIMARY KEY,
        query_example TEXT
    )
    """,
    """
    CREATE TABLE dm_prompts (
        name TEXT PRIMARY KEY,
        text TEXT
    )
    """,
    """
    CREATE TABLE dm_conversation_lifecycle (
        event TEXT PRIMARY KEY,
        text TEXT
    )
    """,
]

INSERT_TEST_DATA_SQL = [
    """
    INSERT INTO dm_anchors (noun, description, id_example, query) VALUES
    ('document', 'document', 'document:doc42', ''),
    ('document_chunk', 'document chunk', 'document_chunk_id:doc42_001',
     'MATCH (d:document_chunk) WHERE ID(d)="document_chunk_id:doc42_001" RETURN d'),
    ('consultation', 'main anchor', '',
     'MATCH (c:consultation) WHERE c.consultation_text CONTAINS "keyword" RETURN c, c.consultation_text'),
    ('faq', 'frequent questions and answers', 'answer:001', '')
    """,
    """
    INSERT INTO dm_links (anchor1, anchor2, sentence, description, query,
                        anchor1_link_column_name, anchor2_link_column_name, has_direction) VALUES
    ('document', 'document_chunk', 'DOCUMENT_has_DOCUMENT_CHUNK', '',
     'MATCH (d:document)-[rel:DOCUMENT_has_DOCUMENT_CHUNK]->(c:document_chunk) RETURN d, c',
     'link_document_has_document_chunk', 'link_document_has_document_chunk', 0),
    ('document_chunk', 'faq', 'DOCUMENT_CHUNK_has_FAQ', '',
     'MATCH (d:document_chunk)-[rel:DOCUMENT_CHUNK_has_FAQ]->(f:faq) RETURN d, f',
     'link_document_chunk_has_faq', 'link_document_chunk_has_faq', 0)
    """,
    """
    INSERT INTO dm_anchor_attributes (anchor, attribute_name, description, data_example,
                                     embeddable, query, dtype, embed_threshold) VALUES
    ('consultation', 'consultation_text', 'text', '', 1,
     'vts_fn(label=''consultation'', property=''consultation_text'', text=''<user question>'')',
     'str', 0.05),
    ('faq', 'faq_question_text', 'extract topic and search using faq_question_text', '', 1,
     'vts_fn(label=''faq'', property=''faq_question_text'', text=''<entity>'')', 'str', 0.01),
    ('faq', 'faq_answer_text', '', '', 1, '', 'str', 0.2)
    """,
    """
    INSERT INTO dm_link_attributes (link, attribute_name, description, data_example,
                                   embeddable, query, dtype, embed_threshold) VALUES
    ('DOCUMENT_has_DOCUMENT_CHUNK', 'chunk_order', 'order of chunk in document', '1', 0,
     '', 'int', 0.0),
    ('DOCUMENT_CHUNK_has_FAQ', 'relevance_score', 'relevance score', '0.85', 1,
     '', 'float', 0.1)
    """,
    """
    INSERT INTO dm_queries (query_name, query_example) VALUES
    ('ANY QUESTION', 'extract topics, search FAQs, then search consultations')
    """,
    """
    INSERT INTO dm_prompts (name, text) VALUES
    ('dm_descr_template',
     '## Nodes:\\n{anchors}\\n\\n## Node attributes:\\n{anchor_attrs}\\n\\n## Links:\\n{links}\\n\\n## Link attributes:\\n{link_attrs}\\n\\n## Frequent questions:\\n{queries}'),
    ('generate_no_answer_tmplt', 'Tell the user we could not find the answer.')
    """,
    """
    INSERT INTO dm_conversation_lifecycle (event, text) VALUES
    ('/start', 'Hello!')
    """,
]


@pytest_asyncio.fixture
async def test_db_engine():
    engine = sa_aio.create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    async with engine.begin() as conn:
        for statement in CREATE_REQUIRED_TABLES_SQL + CREATE_OPTIONAL_TABLES_SQL:
            await conn.execute(sa.text(statement))

    yield engine

    await engine.dispose()


@pytest.fixture
def test_sessionmaker(
    test_db_engine,
) -> sa_aio.async_sessionmaker[sa_aio.AsyncSession]:
    return sa_aio.async_sessionmaker(bind=test_db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def test_data(test_db_engine):
    """Insert test data into the test database."""
    async with test_db_engine.begin() as conn:
        for statement in INSERT_TEST_DATA_SQL:
            await conn.execute(sa.text(statement))

    yield


@pytest.mark.asyncio
async def test_dm_usage(test_sessionmaker, test_data):
    """Test DataModel reading from database."""
    dm = DataModel.create(sessionmaker=test_sessionmaker)

    vis = set(await dm.vector_indices())
    assert ("anchor", "consultation", "consultation_text", 0.05) in vis
    assert ("anchor", "faq", "faq_question_text", 0.01) in vis
    assert ("anchor", "faq", "faq_answer_text", 0.2) in vis
    assert ("edge", "DOCUMENT_CHUNK_has_FAQ", "relevance_score", 0.1) in vis

    doc_links = {link.sentence for link in await dm.anchor_links("document")}
    assert "DOCUMENT_has_DOCUMENT_CHUNK" in doc_links

    tmpls = await dm.prompt_templates()
    assert "dm_descr_template" in tmpls
    assert "generate_no_answer_tmplt" in tmpls

    lifecycle = await dm.conversation_lifecycle_events()
    assert lifecycle.get("/start") == "Hello!"

    text_descr = await dm.to_text_descr()
    assert "consultation_text" in text_descr
    assert "DOCUMENT_has_DOCUMENT_CHUNK" in text_descr


@pytest.mark.asyncio
async def test_dm_anchors(test_sessionmaker, test_data):
    """Test reading anchors from database."""
    dm = DataModel.create(sessionmaker=test_sessionmaker)

    anchors = await dm.get_anchors()
    anchor_nouns = {anchor.noun for anchor in anchors}

    assert len(anchors) == 4
    assert "document" in anchor_nouns
    assert "document_chunk" in anchor_nouns
    assert "consultation" in anchor_nouns
    assert "faq" in anchor_nouns

    consultation_anchor = next(a for a in anchors if a.noun == "consultation")
    assert consultation_anchor.description == "main anchor"

    consultation_attrs = [attr for attr in consultation_anchor.attributes if attr.name == "consultation_text"]
    assert len(consultation_attrs) == 1
    assert consultation_attrs[0].embed_threshold == 0.05

    faq_anchor = next(a for a in anchors if a.noun == "faq")
    assert len(faq_anchor.attributes) == 2
    faq_attr_names = {attr.name for attr in faq_anchor.attributes}
    assert "faq_question_text" in faq_attr_names
    assert "faq_answer_text" in faq_attr_names

    document_anchor = next(a for a in anchors if a.noun == "document")
    assert len(document_anchor.attributes) == 0


@pytest.mark.asyncio
async def test_dm_links(test_sessionmaker, test_data):
    """Test reading links from database."""
    dm = DataModel.create(sessionmaker=test_sessionmaker)

    anchors = await dm.get_anchors()
    links = await dm.get_links(anchors_dict={anchor.noun: anchor for anchor in anchors})
    link_sentences = {link.sentence for link in links}

    assert len(links) == 2
    assert "DOCUMENT_has_DOCUMENT_CHUNK" in link_sentences
    assert "DOCUMENT_CHUNK_has_FAQ" in link_sentences

    doc_link = next(link for link in links if link.sentence == "DOCUMENT_has_DOCUMENT_CHUNK")
    assert doc_link.anchor_from.noun == "document"
    assert doc_link.anchor_to.noun == "document_chunk"
    assert doc_link.has_direction is False

    assert len(doc_link.attributes) == 1
    assert doc_link.attributes[0].name == "chunk_order"
    assert doc_link.attributes[0].dtype == "int"

    faq_link = next(link for link in links if link.sentence == "DOCUMENT_CHUNK_has_FAQ")
    assert len(faq_link.attributes) == 1
    assert faq_link.attributes[0].name == "relevance_score"
    assert faq_link.attributes[0].embeddable is True
    assert faq_link.attributes[0].embed_threshold == 0.1


@pytest.mark.asyncio
async def test_dm_attributes(test_sessionmaker, test_data):
    """Test reading attributes from database."""
    dm = DataModel.create(sessionmaker=test_sessionmaker)

    anchors = await dm.get_anchors()
    links = await dm.get_links(anchors_dict={anchor.noun: anchor for anchor in anchors})

    anchor_attrs = []
    for anchor in anchors:
        anchor_attrs.extend(anchor.attributes)

    anchor_attr_names = {attr.name for attr in anchor_attrs}

    assert len(anchor_attrs) == 3
    assert "consultation_text" in anchor_attr_names
    assert "faq_question_text" in anchor_attr_names
    assert "faq_answer_text" in anchor_attr_names

    consultation_attr = next(a for a in anchor_attrs if a.name == "consultation_text")
    assert consultation_attr.embeddable is True
    assert consultation_attr.embed_threshold == 0.05
    assert consultation_attr.dtype == "str"

    consultation_anchor = next(a for a in anchors if a.noun == "consultation")
    assert any(attr.name == "consultation_text" for attr in consultation_anchor.attributes)

    faq_anchor = next(a for a in anchors if a.noun == "faq")
    assert any(attr.name == "faq_question_text" for attr in faq_anchor.attributes)
    assert any(attr.name == "faq_answer_text" for attr in faq_anchor.attributes)

    link_attrs = []
    for link in links:
        link_attrs.extend(link.attributes)

    link_attr_names = {attr.name for attr in link_attrs}

    assert len(link_attrs) == 2
    assert "chunk_order" in link_attr_names
    assert "relevance_score" in link_attr_names

    doc_link = next(link for link in links if link.sentence == "DOCUMENT_has_DOCUMENT_CHUNK")
    assert any(attr.name == "chunk_order" for attr in doc_link.attributes)

    faq_link = next(link for link in links if link.sentence == "DOCUMENT_CHUNK_has_FAQ")
    assert any(attr.name == "relevance_score" for attr in faq_link.attributes)


@pytest.mark.asyncio
async def test_dm_queries(test_sessionmaker, test_data):
    """Test reading queries from database."""
    dm = DataModel.create(sessionmaker=test_sessionmaker)

    queries = await dm.get_queries()
    assert len(queries) == 1
    assert queries[0].name == "ANY QUESTION"
    assert "extract topics" in queries[0].example


@pytest.mark.asyncio
async def test_dm_prompts(test_sessionmaker, test_data):
    """Test reading prompts from database."""
    dm = DataModel.create(sessionmaker=test_sessionmaker)

    prompts = await dm.get_prompts()
    prompt_names = {p.name for p in prompts}

    assert len(prompts) == 2
    assert "dm_descr_template" in prompt_names
    assert "generate_no_answer_tmplt" in prompt_names

    templates = await dm.prompt_templates()
    assert "dm_descr_template" in templates
    assert "## Nodes:" in templates["dm_descr_template"]


@pytest.mark.asyncio
async def test_dm_conversation_lifecycle(test_sessionmaker, test_data):
    """Test reading conversation lifecycle events from database."""
    dm = DataModel.create(sessionmaker=test_sessionmaker)

    lifecycle = await dm.get_conversation_lifecycle_events()
    assert len(lifecycle) == 1
    assert lifecycle[0].event == "/start"
    assert lifecycle[0].text == "Hello!"

    events = await dm.conversation_lifecycle_events()
    assert events.get("/start") == "Hello!"


@pytest.mark.asyncio
async def test_dm_empty_optional_tables():
    """Test that optional tables return empty lists if they don't exist or are empty."""
    engine = sa_aio.create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    try:
        async with engine.begin() as conn:
            for statement in CREATE_REQUIRED_TABLES_SQL:
                await conn.execute(sa.text(statement))

        sessionmaker = sa_aio.async_sessionmaker(bind=engine, expire_on_commit=False)
        dm = DataModel.create(sessionmaker=sessionmaker)

        assert await dm.get_queries() == []
        assert await dm.get_prompts() == []
        assert await dm.get_conversation_lifecycle_events() == []
    finally:
        await engine.dispose()
