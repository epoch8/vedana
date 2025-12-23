import logging
import re
import secrets
from datetime import date
from hashlib import sha256
from typing import Any, Hashable, Iterator, cast
from unicodedata import normalize
from uuid import UUID

import pandas as pd
from jims_core.llms.llm_provider import LLMProvider
from neo4j import GraphDatabase
from vedana_core.db import get_sessionmaker
from vedana_core.data_model import DataModel, Attribute, Anchor, Link
from vedana_core.data_provider import GristAPIDataProvider, GristCsvDataProvider
from vedana_core.graph import MemgraphGraph
from vedana_core.rag_pipeline import RagPipeline
from vedana_core.settings import VedanaCoreSettings
from vedana_core.settings import settings as core_settings

from vedana_etl.settings import settings as etl_settings

# pd.replace() throws warnings due to type downcasting. Behavior will change only in pandas 3.0
# https://github.com/pandas-dev/pandas/issues/57734
pd.set_option("future.no_silent_downcasting", True)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def is_uuid(val: str):
    try:
        UUID(str(val))
        return True
    except (ValueError, TypeError):
        return False


def clean_str(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = normalize("NFC", text)

    # Replace non-breaking spaces and other space-like Unicode chars with regular space
    text = re.sub(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]", " ", text)

    # Remove zero-width spaces and BOMs
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_data_model() -> Iterator[
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
]:
    loader = GristCsvDataProvider(
        doc_id=core_settings.grist_data_model_doc_id,
        grist_server=core_settings.grist_server_url,
        api_key=core_settings.grist_api_key,
    )

    _links_df = loader.get_table("Links")
    links_df = cast(
        pd.DataFrame,
        _links_df[
            [
                "anchor1",
                "anchor2",
                "sentence",
                "description",
                "query",
                "anchor1_link_column_name",
                "anchor2_link_column_name",
                "has_direction",
            ]
        ],
    )
    assert links_df is not None

    links_df["has_direction"] = _links_df["has_direction"].astype(bool)
    links_df = links_df.dropna(subset=["anchor1", "anchor2", "sentence"], inplace=False)

    anchor_attrs_df = loader.get_table("Anchor_attributes")
    anchor_attrs_df = cast(
        pd.DataFrame,
        anchor_attrs_df[
            [
                "anchor",
                "attribute_name",
                "description",
                "data_example",
                "embeddable",
                "query",
                "dtype",
                "embed_threshold",
            ]
        ],
    )
    anchor_attrs_df["embeddable"] = anchor_attrs_df["embeddable"].astype(bool)
    anchor_attrs_df["embed_threshold"] = anchor_attrs_df["embed_threshold"].astype(float)
    anchor_attrs_df = anchor_attrs_df.dropna(subset=["anchor", "attribute_name"], how="any")

    link_attrs_df = loader.get_table("Link_attributes")
    link_attrs_df = cast(
        pd.DataFrame,
        link_attrs_df[
            [
                "link",
                "attribute_name",
                "description",
                "data_example",
                "embeddable",
                "query",
                "dtype",
                "embed_threshold",
            ]
        ],
    )
    link_attrs_df["embeddable"] = link_attrs_df["embeddable"].astype(bool)
    link_attrs_df["embed_threshold"] = link_attrs_df["embed_threshold"].astype(float)
    link_attrs_df = link_attrs_df.dropna(subset=["link", "attribute_name"], how="any")

    anchors_df = loader.get_table("Anchors")
    anchors_df = cast(
        pd.DataFrame,
        anchors_df[
            [
                "noun",
                "description",
                "id_example",
                "query",
            ]
        ],
    )
    anchors_df = anchors_df.dropna(subset=["noun"], inplace=False)
    anchors_df = anchors_df.astype(str)

    queries_df = loader.get_table("Queries")
    queries_df = cast(pd.DataFrame, queries_df[["query_name", "query_example"]])
    queries_df = queries_df.dropna()
    queries_df = queries_df.astype(str)

    prompts_df = loader.get_table("Prompts")
    prompts_df = cast(pd.DataFrame, prompts_df[["name", "text"]])
    prompts_df = prompts_df.dropna()
    prompts_df = prompts_df.astype(str)

    conversation_lifecycle_df = loader.get_table("ConversationLifecycle")
    conversation_lifecycle_df = cast(pd.DataFrame, conversation_lifecycle_df[["event", "text"]])
    conversation_lifecycle_df = conversation_lifecycle_df.dropna()
    conversation_lifecycle_df = conversation_lifecycle_df.astype(str)

    yield anchors_df, anchor_attrs_df, link_attrs_df, links_df, queries_df, prompts_df, conversation_lifecycle_df


def get_grist_data() -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Fetch all anchors and links from Grist into node/edge tables
    """

    # Build necessary DataModel elements from input tables
    dm_anchors_df, dm_anchor_attrs_df, dm_link_attrs_df, dm_links_df, _q, _p, _cl = next(get_data_model())

    # Anchors
    dm_anchors: dict[str, Anchor] = {}
    for _, a_row in dm_anchors_df.iterrows():
        noun = str(a_row.get("noun")).strip()
        if not noun:
            continue
        dm_anchors[noun] = Anchor(
            noun=noun,
            description=a_row.get("description", ""),
            id_example=a_row.get("id_example", ""),
            query=a_row.get("query", ""),
            attributes=[],
        )

    # Anchor attributes
    for _, attr_row in dm_anchor_attrs_df.iterrows():
        noun = str(attr_row.get("anchor")).strip()
        if not noun or noun not in dm_anchors:
            continue
        dm_anchors[noun].attributes.append(
            Attribute(
                name=attr_row.get("attribute_name", ""),
                description=attr_row.get("description", ""),
                example=attr_row.get("data_example", ""),
                dtype=attr_row.get("dtype", ""),
                query=attr_row.get("query", ""),
                embeddable=bool(attr_row.get("embeddable", False)),
                embed_threshold=float(attr_row.get("embed_threshold", 1.0)),
            )
        )

    # Links
    dm_links: dict[str, Link] = {}
    for _, l_row in dm_links_df.iterrows():
        a1 = str(l_row.get("anchor1")).strip()
        a2 = str(l_row.get("anchor2")).strip()
        if not a1 or not a2 or a1 not in dm_anchors or a2 not in dm_anchors:
            logger.error(f'Link type has invalid anchors "{a1} - {a2}", skipping')
            continue
        dm_links[l_row.get("sentence")] = Link(
            anchor_from=dm_anchors[a1],
            anchor_to=dm_anchors[a2],
            sentence=l_row.get("sentence"),
            description=l_row.get("description", ""),
            query=l_row.get("query", ""),
            attributes=[],
            has_direction=bool(l_row.get("has_direction", False)),
            anchor_from_link_attr_name=l_row.get("anchor1_link_column_name", ""),
            anchor_to_link_attr_name=l_row.get("anchor2_link_column_name", ""),
        )

    # Link attributes
    for _, lattr_row in dm_link_attrs_df.iterrows():
        sent = str(lattr_row.get("link")).strip()
        if sent not in dm_links:
            continue
        dm_links[sent].attributes.append(
            Attribute(
                name=str(lattr_row.get("attribute_name")),
                description=str(lattr_row.get("description", "")),
                example=str(lattr_row.get("data_example", "")),
                dtype=str(lattr_row.get("dtype", "")),
                query=str(lattr_row.get("query", "")),
                embeddable=bool(lattr_row.get("embeddable", False)),
                embed_threshold=float(lattr_row.get("embed_threshold", 1.0)),
            )
        )

    # Get data from Grist

    dp = GristAPIDataProvider(
        doc_id=core_settings.grist_data_doc_id,
        grist_server=core_settings.grist_server_url,
        api_key=core_settings.grist_api_key,
    )

    # Foreign key type links
    fk_link_records_from = []
    fk_link_records_to = []

    # Nodes
    node_records: dict[str, Any] = {}
    anchor_types = dp.get_anchor_tables()  # does not check data model! only lists tables that are named anchor_...
    logger.debug(f"Fetching {len(anchor_types)} anchor tables from Grist: {anchor_types}")

    for anchor_type in anchor_types:
        # check anchor's existence in data model
        dm_anchor = dm_anchors.get(anchor_type)
        if not dm_anchor:
            logger.error(f'Anchor "{anchor_type}" not described in data model, skipping')
            continue
        dm_anchor_attrs = [attr.name for attr in dm_anchor.attributes]

        # get anchor's links
        # todo check link column directions
        anchor_from_link_cols = [link for link in dm_links.values() if link.anchor_from.noun == anchor_type and link.anchor_from_link_attr_name]
        anchor_to_link_cols = [link for link in dm_links.values() if link.anchor_to.noun == anchor_type and link.anchor_to_link_attr_name]

        try:
            anchors = dp.get_anchors(anchor_type, dm_attrs=dm_anchor.attributes, dm_anchor_links=anchor_from_link_cols)
        except Exception as exc:
            logger.exception(f"Failed to fetch anchors for type {anchor_type}: {exc}")
            continue

        node_records[anchor_type] = {}

        for a in anchors:
            for link in anchor_from_link_cols:
                # get link other end id(s)
                link_ids = a.data.get(link.anchor_from_link_attr_name, [])
                if isinstance(link_ids, (int, str)):
                    link_ids = [link_ids]
                elif link_ids is None:
                    continue

                for to_dp_id in link_ids:
                    fk_link_records_from.append(
                        {
                            "from_node_id": a.id,
                            "from_node_dp_id": a.dp_id,
                            # "to_node_id": link.id_to, <-- not provided here
                            "to_node_dp_id": to_dp_id,
                            "from_node_type": anchor_type,
                            "to_node_type": link.anchor_to.noun,
                            "edge_label": link.sentence,
                            # "attributes": {},  # not present in data (format not specified) yet
                        }
                    )

            for link in anchor_to_link_cols:
                # get link other end id(s)
                link_ids = a.data.get(link.anchor_to_link_attr_name, [])
                if isinstance(link_ids, (int, str)):
                    link_ids = [link_ids]
                elif link_ids is None:
                    continue

                for from_dp_id in link_ids:
                    fk_link_records_to.append(
                        {
                            # "from_node_id": link.id_from,  <-- not provided here
                            "from_node_dp_id": from_dp_id,
                            "to_node_id": a.id,
                            "to_node_dp_id": a.dp_id,
                            "from_node_type": link.anchor_from.noun,
                            "to_node_type": anchor_type,
                            "edge_label": link.sentence,
                            # "attributes": {},  # not present in data (format not specified) yet
                        }
                    )

            node_records[anchor_type][a.dp_id] = {
                "node_id": a.id,
                "node_type": a.type,
                "attributes": {k: v for k, v in a.data.items() if k in dm_anchor_attrs} or {},
            }

    nodes_df = pd.DataFrame(
        [
            {"node_id": rec.get("node_id"), "node_type": rec.get("node_type"), "attributes": rec.get("attributes", {})}
            for a in node_records.values()
            for rec in a.values()
        ],
        columns=["node_id", "node_type", "attributes"],
    )

    # Resolve links (database id <-> our id), if necessary
    for lk in fk_link_records_to:
        if isinstance(lk["from_node_dp_id"], int):
            lk["from_node_id"] = node_records[lk["from_node_type"]].get(lk["from_node_dp_id"], {}).get("node_id")
        else:
            lk["from_node_id"] = lk["from_node_dp_id"]  # <-- str dp_id is an already correct id
    for lk in fk_link_records_from:
        if isinstance(lk["to_node_dp_id"], int):
            lk["to_node_id"] = node_records[lk["to_node_type"]].get(lk["to_node_dp_id"], {}).get("node_id")
        else:
            lk["to_node_id"] = lk["to_node_dp_id"]

    if fk_link_records_to:
        fk_links_to_df = pd.DataFrame(fk_link_records_to).dropna(subset=["from_node_id", "to_node_id"])
    else:
        fk_links_to_df = pd.DataFrame(
            columns=["from_node_id", "to_node_id", "from_node_type", "to_node_type", "edge_label"]
        )

    if fk_link_records_from:
        fk_links_from_df = pd.DataFrame(fk_link_records_from).dropna(subset=["from_node_id", "to_node_id"])
    else:
        fk_links_from_df = pd.DataFrame(
            columns=["from_node_id", "to_node_id", "from_node_type", "to_node_type", "edge_label"]
        )

    fk_df = pd.concat([fk_links_from_df, fk_links_to_df], axis=0, ignore_index=True)
    fk_df["attributes"] = [dict()] * fk_df.shape[0]
    fk_df = fk_df[["from_node_id", "to_node_id", "from_node_type", "to_node_type", "edge_label", "attributes"]]

    # keep only links with both nodes present (+done in the end on edges_df); todo add test for this case
    fk_df = fk_df.loc[(fk_df["from_node_id"].isin(nodes_df["node_id"]) & fk_df["to_node_id"].isin(nodes_df["node_id"]))]

    # Edges
    edge_records = []
    link_types = dp.get_link_tables()
    logger.debug(f"Fetching {len(link_types)} link types from Grist: {link_types}")

    for link_type in link_types:
        # check link's existence in data model (dm_link is used from anchor_from / to references only)
        dm_link_list = [
            link
            for link in dm_links.values()
            if link.sentence.lower() == link_type.lower()
            or link_type.lower() == f"{link.anchor_from.noun}_{link.anchor_to.noun}".lower()
        ]
        if not dm_link_list:
            logger.error(f'Link type "{link_type}" not described in data model, skipping')
            continue
        dm_link = dm_link_list[0]
        dm_link_attrs = [a.name for a in dm_link.attributes]

        try:
            links = dp.get_links(link_type, dm_link)
        except Exception as exc:
            logger.error(f"Failed to fetch links for type {link_type}: {exc}")
            continue

        for link_record in links:
            id_from = link_record.id_from
            id_to = link_record.id_to

            # resolve foreign key link_record id's
            if isinstance(id_from, int):
                id_from = node_records[link_record.anchor_from].get(id_from, {}).get("node_id")
            if isinstance(id_to, int):
                id_to = node_records[link_record.anchor_to].get(id_to, {}).get("node_id")

            edge_records.append(
                {
                    "from_node_id": id_from,
                    "to_node_id": id_to,
                    "from_node_type": link_record.anchor_from,
                    "to_node_type": link_record.anchor_to,
                    "edge_label": link_record.type,
                    "attributes": {k: v for k, v in link_record.data.items() if k in dm_link_attrs} or {},
                }
            )

    edges_df = pd.DataFrame(edge_records)
    edges_df = edges_df.loc[
        (edges_df["from_node_id"].isin(nodes_df["node_id"]) & edges_df["to_node_id"].isin(nodes_df["node_id"]))
    ]

    edges_df = pd.concat([edges_df, fk_df], ignore_index=True)

    # add reverse links (if already provided in data, duplicates will be removed later)
    for link in dm_links.values():
        if not link.has_direction:
            rev_edges = cast(
                pd.DataFrame,
                edges_df.loc[
                    (
                        (
                            (edges_df["from_node_type"] == link.anchor_from.noun)
                            & (edges_df["to_node_type"] == link.anchor_to.noun)
                        )
                        | (  # edges with anchors written in reverse are also valid
                            (edges_df["from_node_type"] == link.anchor_to.noun)
                            & (edges_df["to_node_type"] == link.anchor_from.noun)
                        )
                    )
                    & (edges_df["edge_label"] == link.sentence)
                ].copy(),
            )
            if not rev_edges.empty:
                rev_edges = rev_edges.rename(
                    columns={
                        "from_node_id": "to_node_id",
                        "to_node_id": "from_node_id",
                        "from_node_type": "to_node_type",
                        "to_node_type": "from_node_type",
                    }
                )
                edges_df = pd.concat([edges_df, rev_edges], ignore_index=True)

    # preventive drop_duplicates / na records
    if not nodes_df.empty:
        nodes_df = nodes_df.dropna(subset=["node_id", "node_type"]).drop_duplicates(subset=["node_id"])
    if not edges_df.empty:
        edges_df = edges_df.dropna(subset=["from_node_id", "to_node_id", "edge_label"]).drop_duplicates(
            subset=["from_node_id", "to_node_id", "edge_label"]
        )
    yield nodes_df, edges_df


def ensure_memgraph_node_indexes(dm_anchor_attrs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create label / vector indices
    https://memgraph.com/docs/querying/vector-search
    """

    anchor_types: set[str] = set(dm_anchor_attrs["anchor"].dropna().unique())

    # embeddable attrs for vector indices
    vec_a_attr_rows = dm_anchor_attrs[dm_anchor_attrs["embeddable"]]  #  & (dm_anchor_attrs["dtype"].str.lower() == "str")]

    driver = GraphDatabase.driver(
        uri=core_settings.memgraph_uri,
        auth=(
            core_settings.memgraph_user,
            core_settings.memgraph_pwd,
        ),
    )

    with driver.session() as session:
        # Indices on Anchors
        for label in anchor_types:
            try:
                session.run(f"CREATE INDEX ON :`{label}`(id)")  # type: ignore
            except Exception as exc:
                logger.debug(f"CREATE INDEX failed for label {label}: {exc}")  # probably index exists

            try:
                session.run(f"CREATE CONSTRAINT ON (n:`{label}`) ASSERT n.id IS UNIQUE")  # type: ignore
            except Exception as exc:
                logger.debug(f"CREATE CONSTRAINT failed for label {label}: {exc}")  # probably index exists

        # Vector indices - Anchors
        for _, row in vec_a_attr_rows.iterrows():
            attr: str = row["attribute_name"]
            embeddings_dim = core_settings.embeddings_dim
            label = row["anchor"]

            idx_name = f"{label}_{attr}_embed_idx".replace(" ", "_")
            prop_name = f"{attr}_embedding"

            cypher = (
                f"CREATE VECTOR INDEX `{idx_name}` ON :`{label}`(`{prop_name}`) "
                f'WITH CONFIG {{"dimension": {embeddings_dim}, "capacity": 1024, "metric": "cos"}}'
            )
            try:
                session.run(cypher)  # type: ignore
            except Exception as exc:
                logger.debug(f"CREATE VECTOR INDEX failed for {idx_name}: {exc}")  # probably index exists
                continue

    driver.close()

    # nominal outputs
    mg_anchor_indexes = pd.DataFrame({"anchor": list(anchor_types)})
    mg_anchor_vector_indexes = vec_a_attr_rows[["anchor", "attribute_name"]].copy()
    return mg_anchor_indexes, mg_anchor_vector_indexes


def ensure_memgraph_edge_indexes(dm_link_attrs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create label / vector indices
    https://memgraph.com/docs/querying/vector-search
    """

    link_types: set[str] = set(dm_link_attrs["link"].dropna().unique())

    # embeddable attrs for vector indices
    vec_l_attr_rows = dm_link_attrs[dm_link_attrs["embeddable"]]  # & (dm_link_attrs["dtype"].str.lower() == "str")]

    driver = GraphDatabase.driver(
        uri=core_settings.memgraph_uri,
        auth=(
            core_settings.memgraph_user,
            core_settings.memgraph_pwd,
        ),
    )

    with driver.session() as session:
        # Indices on Edges (optimizes queries such as MATCH ()-[r:EDGE_TYPE]->() RETURN r;)
        # If queried by edge property, will need to add property index (similar to above for Anchor)
        for label in link_types:
            try:
                session.run(f"CREATE EDGE INDEX ON :`{label}`")  # type: ignore
            except Exception as exc:
                logger.debug(f"CREATE EDGE INDEX failed for label {label}: {exc}")  # probably index exists

            # todo edge constraints?
            # try:
            #     session.run(f"CREATE CONSTRAINT ON (n:`{label}`) ASSERT n.id IS UNIQUE")  # type: ignore
            # except Exception as exc:
            #     logger.debug(f"CREATE CONSTRAINT failed for label {label}: {exc}")  # probably index exists

        # Vector indices
        for _, row in vec_l_attr_rows.iterrows():
            attr: str = row["attribute_name"]
            embeddings_dim = core_settings.embeddings_dim
            label = row["link"]

            idx_name = f"{label}_{attr}_embed_idx".replace(" ", "_")
            prop_name = f"{attr}_embedding"

            cypher = (
                f"CREATE VECTOR EDGE INDEX `{idx_name}` ON :`{label}`(`{prop_name}`) "
                f'WITH CONFIG {{"dimension": {embeddings_dim}, "capacity": 1024, "metric": "cos"}}'
            )
            try:
                session.run(cypher)  # type: ignore
            except Exception as exc:
                logger.debug(f"CREATE VECTOR EDGE INDEX failed for {idx_name}: {exc}")  # probably index exists
                continue

    driver.close()

    # nominal outputs
    mg_link_indexes = pd.DataFrame({"link": list(link_types)})
    mg_link_vector_indexes = vec_l_attr_rows[["link", "attribute_name"]].copy()
    return mg_link_indexes, mg_link_vector_indexes


def generate_embeddings(
    df: pd.DataFrame,
    memgraph_vector_indexes: pd.DataFrame,
) -> pd.DataFrame:
    """Generate embeddings for embeddable text attributes"""

    if df.empty:
        return df

    type_col = "node_type" if "node_id" in df.columns else "edge_label"

    # Build mapping type -> list[attribute_name] that need embedding
    mapping: dict[str, list[str]] = {}
    for _, row in memgraph_vector_indexes.iterrows():
        record_type = row["anchor"] if type_col == "node_type" else row["link"]
        if pd.isna(record_type):
            continue
        mapping.setdefault(record_type, []).append(row["attribute_name"])

    tasks: list[tuple[Hashable, str, str]] = []  # (row_idx, attr_name, text)

    for idx, row in df.iterrows():
        typ_val = row[type_col]
        attrs_needed = mapping.get(typ_val)
        if not attrs_needed:
            continue
        attr_dict = row["attributes"] or {}
        for attr_name in attrs_needed:
            text_val = attr_dict.get(attr_name)
            if text_val and isinstance(text_val, str) and not is_uuid(text_val):
                tasks.append((idx, attr_name, text_val))

    if not tasks:
        return df

    provider = LLMProvider()

    texts = [t[2] for t in tasks]
    vectors = provider.create_embeddings_sync(texts)

    # Re-init attributes to store only embeddings
    # df = df.drop(columns=["attributes"])
    # df["attributes"] = pd.NA

    # Apply embeddings to df
    for (row_idx, attr_name, _), vec in zip(tasks, vectors):
        attr_dict = df.at[row_idx, "attributes"]
        if pd.isna(attr_dict):
            attr_dict = {}
        else:
            attr_dict = dict(attr_dict)
        attr_dict[f"{attr_name}_embedding"] = vec
        df.at[row_idx, "attributes"] = attr_dict

    # remove rows without embeddings
    # df = df.dropna(subset=["attributes"])

    return df


def merge_attr_dicts(dicts):
    result = {}
    for d in dicts:
        result.update(d)
    return result


# ---
# Custom parts


def prepare_nodes(
    grist_nodes_df: pd.DataFrame,
) -> pd.DataFrame:
    return grist_nodes_df.copy()


def prepare_edges(
    grist_edges_df: pd.DataFrame,
) -> pd.DataFrame:
    return grist_edges_df.copy()


def get_eval_gds_from_grist() -> Iterator[pd.DataFrame]:
    """
    Loads evaluation dataset and config rows

    Output:
      - eval_gds(gds_question, gds_answer, question_context)
    """
    dp = GristAPIDataProvider(
        doc_id=etl_settings.grist_test_set_doc_id,
        grist_server=core_settings.grist_server_url,
        api_key=core_settings.grist_api_key,
    )

    try:
        gds_df = dp.get_table(etl_settings.gds_table_name)
    except Exception as e:
        logger.exception(f"Failed to get golden dataset {etl_settings.gds_table_name}: {e}")
        raise e

    gds_df = gds_df.dropna(subset=["gds_question", "gds_answer"]).copy()
    gds_df = gds_df.loc[(gds_df["gds_question"] != "") & (gds_df["gds_answer"] != "")]
    gds_df = gds_df[["gds_question", "gds_answer", "question_scenario", "question_comment", "question_context"]].astype(
        {"gds_question": str, "gds_answer": str, "question_scenario": str, "question_comment": str}
    )
    yield gds_df


if __name__ == "__main__":
    # import dotenv
    # dotenv.load_dotenv()
    # n, _l = next(get_grist_data())
    ...
