import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine

from vedana_core.data_model import DataModel


@pytest.fixture
def test_db_engine():
    engine = create_engine("sqlite:///:memory:")

    # Create tables
    with engine.connect() as conn:
        conn.execute(
            sa.text("""
                CREATE TABLE dm_anchors (
                    noun TEXT PRIMARY KEY,
                    description TEXT,
                    id_example TEXT,
                    query TEXT
                )
            """)
        )

        conn.execute(
            sa.text("""
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
            """)
        )

        conn.execute(
            sa.text("""
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
            """)
        )

        conn.execute(
            sa.text("""
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
            """)
        )

        conn.execute(
            sa.text("""
                CREATE TABLE dm_queries (
                    name TEXT PRIMARY KEY,
                    example TEXT
                )
            """)
        )

        conn.execute(
            sa.text("""
                CREATE TABLE dm_prompts (
                    name TEXT PRIMARY KEY,
                    text TEXT
                )
            """)
        )

        conn.execute(
            sa.text("""
                CREATE TABLE dm_conversation_lifecycle (
                    event TEXT PRIMARY KEY,
                    text TEXT
                )
            """)
        )

        conn.commit()

    yield engine
    engine.dispose()


@pytest.fixture
def test_data(test_db_engine):
    """Insert test data into the test database."""
    with test_db_engine.connect() as conn:
        # Insert anchors
        conn.execute(
            sa.text("""
                INSERT INTO dm_anchors (noun, description, id_example, query) VALUES
                ('document', 'document', 'document:doc42', ''),
                ('document_chunk', 'document chunk', 'document_chunk_id:doc42_001', 
                 'MATCH (d:document_chunk) WHERE ID(d)="document_chunk_id:doc42_001" RETURN d'),
                ('consultation', 'main anchor', '', 
                 'MATCH (c:consultation) WHERE c.consultation_text CONTAINS "keyword" RETURN c, c.consultation_text'),
                ('faq', 'frequent questions and answers', 'answer:001', '')
            """)
        )

        # Insert links
        conn.execute(
            sa.text("""
                INSERT INTO dm_links (anchor1, anchor2, sentence, description, query, 
                                    anchor1_link_column_name, anchor2_link_column_name, has_direction) VALUES
                ('document', 'document_chunk', 'DOCUMENT_has_DOCUMENT_CHUNK', '', 
                 'MATCH (d:document)-[rel:DOCUMENT_has_DOCUMENT_CHUNK]->(c:document_chunk) RETURN d, c',
                 'link_document_has_document_chunk', 'link_document_has_document_chunk', 0),
                ('document_chunk', 'faq', 'DOCUMENT_CHUNK_has_FAQ', '', 
                 'MATCH (d:document_chunk)-[rel:DOCUMENT_CHUNK_has_FAQ]->(f:faq) RETURN d, f',
                 'link_document_chunk_has_faq', 'link_document_chunk_has_faq', 0)
            """)
        )

        # Insert anchor attributes
        conn.execute(
            sa.text("""
                INSERT INTO dm_anchor_attributes (anchor, attribute_name, description, data_example, 
                                                 embeddable, query, dtype, embed_threshold) VALUES
                ('consultation', 'consultation_text', 'text', '', 1, 
                 'vts_fn(label=''consultation'', property=''consultation_text'', text=''<user question>'')', 
                 'str', 0.05),
                ('faq', 'faq_question_text', 'extract topic and search using faq_question_text', '', 1, 
                 'vts_fn(label=''faq'', property=''faq_question_text'', text=''<entity>'')', 'str', 0.01),
                ('faq', 'faq_answer_text', '', '', 1, '', 'str', 0.2)
            """)
        )

        # Insert link attributes
        conn.execute(
            sa.text("""
                INSERT INTO dm_link_attributes (link, attribute_name, description, data_example, 
                                               embeddable, query, dtype, embed_threshold) VALUES
                ('DOCUMENT_has_DOCUMENT_CHUNK', 'chunk_order', 'order of chunk in document', '1', 0, 
                 '', 'int', 0.0),
                ('DOCUMENT_CHUNK_has_FAQ', 'relevance_score', 'relevance score', '0.85', 1, 
                 '', 'float', 0.1)
            """)
        )

        # Insert queries
        conn.execute(
            sa.text("""
                INSERT INTO dm_queries (name, example) VALUES
                ('ANY QUESTION', 'extract topics, search FAQs, then search consultations')
            """)
        )

        # Insert prompts
        conn.execute(
            sa.text("""
                INSERT INTO dm_prompts (name, text) VALUES
                ('dm_descr_template', 
                 '## Nodes:\\n{anchors}\\n\\n## Node attributes:\\n{anchor_attrs}\\n\\n## Links:\\n{links}\\n\\n## Link attributes:\\n{link_attrs}\\n\\n## Frequent questions:\\n{queries}'),
                ('generate_no_answer_tmplt', 'Tell the user we could not find the answer.')
            """)
        )

        # Insert conversation lifecycle
        conn.execute(
            sa.text("""
                INSERT INTO dm_conversation_lifecycle (event, text) VALUES
                ('/start', 'Hello!')
            """)
        )

        conn.commit()


def test_dm_usage(test_db_engine, test_data):
    """Test DataModel reading from database."""
    dm = DataModel.create(db_engine=test_db_engine)

    # Vector indices used by agent tools (label, property)
    vis = set(dm.vector_indices())
    assert ("anchor", "consultation", "consultation_text") in vis
    assert ("anchor", "faq", "faq_question_text") in vis
    assert ("anchor", "faq", "faq_answer_text") in vis
    assert ("edge", "DOCUMENT_CHUNK_has_FAQ", "relevance_score") in vis

    # Anchor links (foreign key style hints)
    doc_links = {link.sentence for link in dm.anchor_links("document")}
    assert "DOCUMENT_has_DOCUMENT_CHUNK" in doc_links

    # Prompt templates and lifecycle events used by pipeline
    tmpls = dm.prompt_templates()
    assert "dm_descr_template" in tmpls
    assert "generate_no_answer_tmplt" in tmpls
    lifecycle = dm.conversation_lifecycle_events()
    assert lifecycle.get("/start") == "Hello!"

    # Human-readable description is populated with core parts
    text_descr = dm.to_text_descr()
    assert "consultation_text" in text_descr
    assert "DOCUMENT_has_DOCUMENT_CHUNK" in text_descr


def test_dm_anchors(test_db_engine, test_data):
    """Test reading anchors from database."""
    dm = DataModel.create(db_engine=test_db_engine)

    anchors = dm.anchors
    anchor_nouns = {anchor.noun for anchor in anchors}

    assert len(anchors) == 4
    assert "document" in anchor_nouns
    assert "document_chunk" in anchor_nouns
    assert "consultation" in anchor_nouns
    assert "faq" in anchor_nouns

    # Check specific anchor details
    consultation_anchor = next(a for a in anchors if a.noun == "consultation")
    assert consultation_anchor.description == "main anchor"

    # Check anchor attributes are loaded correctly
    consultation_attrs = [attr for attr in consultation_anchor.attributes if attr.name == "consultation_text"]
    assert len(consultation_attrs) == 1
    assert consultation_attrs[0].embed_threshold == 0.05

    # Check faq anchor has multiple attributes
    faq_anchor = next(a for a in anchors if a.noun == "faq")
    assert len(faq_anchor.attributes) == 2
    faq_attr_names = {attr.name for attr in faq_anchor.attributes}
    assert "faq_question_text" in faq_attr_names
    assert "faq_answer_text" in faq_attr_names

    # Check anchors without attributes are included
    document_anchor = next(a for a in anchors if a.noun == "document")
    assert len(document_anchor.attributes) == 0


def test_dm_links(test_db_engine, test_data):
    """Test reading links from database."""
    dm = DataModel.create(db_engine=test_db_engine)

    links = dm.links
    link_sentences = {link.sentence for link in links}

    assert len(links) == 2
    assert "DOCUMENT_has_DOCUMENT_CHUNK" in link_sentences
    assert "DOCUMENT_CHUNK_has_FAQ" in link_sentences

    # Check link details
    doc_link = next(l for l in links if l.sentence == "DOCUMENT_has_DOCUMENT_CHUNK")
    assert doc_link.anchor_from.noun == "document"
    assert doc_link.anchor_to.noun == "document_chunk"
    assert doc_link.has_direction is False

    # Check link attributes are loaded correctly
    assert len(doc_link.attributes) == 1
    assert doc_link.attributes[0].name == "chunk_order"
    assert doc_link.attributes[0].dtype == "int"

    # Check second link has attributes
    faq_link = next(l for l in links if l.sentence == "DOCUMENT_CHUNK_has_FAQ")
    assert len(faq_link.attributes) == 1
    assert faq_link.attributes[0].name == "relevance_score"
    assert faq_link.attributes[0].embeddable is True
    assert faq_link.attributes[0].embed_threshold == 0.1


def test_dm_attributes(test_db_engine, test_data):
    """Test reading attributes from database."""
    dm = DataModel.create(db_engine=test_db_engine)

    anchors = dm.anchors
    links = dm.links

    # Collect all anchor attributes
    anchor_attrs = []
    for anchor in anchors:
        anchor_attrs.extend(anchor.attributes)

    anchor_attr_names = {attr.name for attr in anchor_attrs}

    assert len(anchor_attrs) == 3
    assert "consultation_text" in anchor_attr_names
    assert "faq_question_text" in anchor_attr_names
    assert "faq_answer_text" in anchor_attr_names

    # Check attribute details and associations
    consultation_attr = next(a for a in anchor_attrs if a.name == "consultation_text")
    assert consultation_attr.embeddable is True
    assert consultation_attr.embed_threshold == 0.05
    assert consultation_attr.dtype == "str"

    # Verify attributes are associated with correct anchors
    consultation_anchor = next(a for a in anchors if a.noun == "consultation")
    assert any(attr.name == "consultation_text" for attr in consultation_anchor.attributes)

    faq_anchor = next(a for a in anchors if a.noun == "faq")
    assert any(attr.name == "faq_question_text" for attr in faq_anchor.attributes)
    assert any(attr.name == "faq_answer_text" for attr in faq_anchor.attributes)

    # Collect all link attributes
    link_attrs = []
    for link in links:
        link_attrs.extend(link.attributes)

    link_attr_names = {attr.name for attr in link_attrs}

    assert len(link_attrs) == 2
    assert "chunk_order" in link_attr_names
    assert "relevance_score" in link_attr_names

    # Verify link attributes are associated with correct links
    doc_link = next(l for l in links if l.sentence == "DOCUMENT_has_DOCUMENT_CHUNK")
    assert any(attr.name == "chunk_order" for attr in doc_link.attributes)

    faq_link = next(l for l in links if l.sentence == "DOCUMENT_CHUNK_has_FAQ")
    assert any(attr.name == "relevance_score" for attr in faq_link.attributes)


def test_dm_queries(test_db_engine, test_data):
    """Test reading queries from database."""
    dm = DataModel.create(db_engine=test_db_engine)

    queries = dm.queries
    assert len(queries) == 1
    assert queries[0].name == "ANY QUESTION"
    assert "extract topics" in queries[0].example


def test_dm_prompts(test_db_engine, test_data):
    """Test reading prompts from database."""
    dm = DataModel.create(db_engine=test_db_engine)

    prompts = dm.prompts
    prompt_names = {p.name for p in prompts}

    assert len(prompts) == 2
    assert "dm_descr_template" in prompt_names
    assert "generate_no_answer_tmplt" in prompt_names

    templates = dm.prompt_templates()
    assert "dm_descr_template" in templates
    assert "## Nodes:" in templates["dm_descr_template"]


def test_dm_conversation_lifecycle(test_db_engine, test_data):
    """Test reading conversation lifecycle events from database."""
    dm = DataModel.create(db_engine=test_db_engine)

    lifecycle = dm.conversation_lifecycle
    assert len(lifecycle) == 1
    assert lifecycle[0].event == "/start"
    assert lifecycle[0].text == "Hello!"

    events = dm.conversation_lifecycle_events()
    assert events.get("/start") == "Hello!"


def test_dm_empty_optional_tables(test_db_engine):
    """Test that optional tables return empty lists if they don't exist or are empty."""
    # Create only required tables
    with test_db_engine.connect() as conn:
        conn.execute(
            sa.text("""
                CREATE TABLE dm_anchors (
                    noun TEXT PRIMARY KEY,
                    description TEXT,
                    id_example TEXT,
                    query TEXT
                )
            """)
        )
        conn.execute(
            sa.text("""
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
            """)
        )
        conn.execute(
            sa.text("""
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
            """)
        )

        conn.execute(
            sa.text("""
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
            """)
        )
        conn.commit()

    dm = DataModel.create(db_engine=test_db_engine)

    # Optional tables should return empty lists
    assert dm.queries == []
    assert dm.prompts == []
    assert dm.conversation_lifecycle == []
