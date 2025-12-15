import pytest
from typing import cast

from vedana_core.graph import Graph
from vedana_core.data_model import (
    Anchor,
    Attribute,
    ConversationLifecycleEvent,
    DataModel,
    Link,
    Prompt,
    Query,
)


class MockGraph:
    """
    Minimal graph stub
    - run_cypher stores JSON content
    - execute_ro_cypher_query returns stored JSON content
    """

    def __init__(self) -> None:
        self.last_params: dict | None = None

    async def run_cypher(self, query: str, params: dict) -> None:
        self.last_params = params

    async def execute_ro_cypher_query(self, query: str):
        content = (self.last_params or {}).get("content")
        return [{"content": content}]


def build_dm() -> DataModel:
    # Anchors
    document = Anchor(
        noun="document",
        description="document",
        id_example="document:doc42",
        query="",
        attributes=[],
    )
    document_chunk = Anchor(
        noun="document_chunk",
        description="document chunk",
        id_example="document_chunk_id:doc42_001",
        query='MATCH (d:document_chunk) WHERE ID(d)="document_chunk_id:doc42_001" RETURN d',
        attributes=[],
    )
    consultation = Anchor(
        noun="consultation",
        description="main anchor",
        id_example="",
        query='MATCH (c:consultation) WHERE c.consultation_text CONTAINS "keyword" RETURN c, c.consultation_text',
        attributes=[],
    )
    faq = Anchor(
        noun="faq",
        description="frequent questions and answers",
        id_example="answer:001",
        query="",
        attributes=[],
    )

    # Links
    link_doc_has_chunk = Link(
        anchor_from=document,
        anchor_to=document_chunk,
        sentence="DOCUMENT_has_DOCUMENT_CHUNK",
        description="",
        query="MATCH (d:document)-[rel:DOCUMENT_has_DOCUMENT_CHUNK]->(c:document_chunk) RETURN d, c",
        attributes=[],
        has_direction=False,
        anchor_from_link_attr_name="link_document_has_document_chunk",
        anchor_to_link_attr_name="link_document_has_document_chunk",
    )

    link_chunk_has_faq = Link(
        anchor_from=document_chunk,
        anchor_to=faq,
        sentence="DOCUMENT_CHUNK_has_FAQ",
        description="",
        query="MATCH (d:document_chunk)-[rel:DOCUMENT_CHUNK_has_FAQ]->(f:faq) RETURN d, f",
        attributes=[],
        has_direction=False,
        anchor_from_link_attr_name="link_document_chunk_has_faq",
        anchor_to_link_attr_name="link_document_chunk_has_faq",
    )

    # Attributes (node)
    a_consultation_text = Attribute(
        name="consultation_text",
        description="text",
        example="",
        query="vts_fn(label='consultation', property='consultation_text', text='<user question>')",
        dtype="str",
        embeddable=True,
        embed_threshold=0.05,
        meta={},
    )
    a_faq_q = Attribute(
        name="faq_question_text",
        description="extract topic and search using faq_question_text",
        example="",
        query='vts_fn(label="faq", property="faq_question_text", text="<entity>")',
        dtype="str",
        embeddable=True,
        embed_threshold=0.01,
        meta={},
    )
    a_faq_a = Attribute(
        name="faq_answer_text",
        description="",
        example="",
        query="",
        dtype="str",
        embeddable=True,
        embed_threshold=0.2,
        meta={},
    )

    consultation.attributes.extend([a_consultation_text])
    faq.attributes.extend([a_faq_q, a_faq_a])

    # Queries
    q_any = Query(
        name="ANY QUESTION",
        example="extract topics, search FAQs, then search consultations",
    )

    # Prompts
    prompts = [
        Prompt(
            name="dm_descr_template",
            text=(
                "## Nodes:\n{anchors}\n\n## Node attributes:\n{anchor_attrs}\n\n## Links:\n{links}\n\n"
                "## Link attributes:\n{link_attrs}\n\n## Frequent questions:\n{queries}"
            ),
        ),
        Prompt(
            name="generate_no_answer_tmplt",
            text="Tell the user we could not find the answer.",
        ),
    ]

    lifecycle = [ConversationLifecycleEvent(event="/start", text="Hello!")]

    return DataModel(
        anchors=[document, document_chunk, consultation, faq],
        links=[link_doc_has_chunk, link_chunk_has_faq],
        attrs=[a_consultation_text, a_faq_q, a_faq_a],
        queries=[q_any],
        conversation_lifecycle=lifecycle,
        prompts=prompts,
    )


def test_dm_usage() -> None:
    dm = build_dm()

    # Embeddable attributes mapping and thresholds
    emb = dm.embeddable_attributes()
    assert emb["consultation_text"]["noun"] == "consultation"
    assert emb["consultation_text"]["th"] == 0.05
    assert emb["faq_question_text"]["noun"] == "faq"
    assert emb["faq_question_text"]["th"] == 0.01
    assert emb["faq_answer_text"]["noun"] == "faq"
    assert emb["faq_answer_text"]["th"] == 0.2

    # Vector indices used by agent tools (label, property)
    vis = set(dm.vector_indices())
    assert ("consultation", "consultation_text") in vis
    assert ("faq", "faq_question_text") in vis
    assert ("faq", "faq_answer_text") in vis

    # Anchor links (foreign key style hints)
    doc_links = {link.sentence for link in dm.anchor_links("document")}
    assert "DOCUMENT_has_DOCUMENT_CHUNK" in doc_links

    # Prompt templates and lifecycle events used by pipeline
    tmpls = dm.prompt_templates()
    assert "dm_descr_template" in tmpls
    lifecycle = dm.conversation_lifecycle_events()
    assert lifecycle.get("/start") == "Hello!"

    # Human-readable description is populated with core parts
    text_descr = dm.to_text_descr()
    assert "consultation_text" in text_descr
    assert "DOCUMENT_has_DOCUMENT_CHUNK" in text_descr


@pytest.mark.asyncio
async def test_dm_graph_cache() -> None:
    dm = build_dm()
    mock_graph = MockGraph()

    await dm.update_data_model_node(cast(Graph, mock_graph))
    assert mock_graph.last_params and isinstance(mock_graph.last_params.get("content"), str)

    dm_loaded = await DataModel.load_from_graph(cast(Graph, mock_graph))
    assert dm_loaded is not None

    # Compare semantic equality via JSON
    assert dm_loaded.to_json() == dm.to_json()
