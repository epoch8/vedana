import logging
import re
from pathlib import Path
from unicodedata import normalize
from uuid import UUID

import pandas as pd
from neo4j import GraphDatabase
from vedana_core.data_provider import GristOnlineCsvDataProvider, GristSQLDataProvider
from vedana_core.embeddings import OpenaiEmbeddingProvider
from vedana_core.settings import settings as core_settings

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


def get_data_model():
    """
    TODO: replace with DataModel from memgraph-rag.data_model once it is refactored as a package.
    """
    loader = GristOnlineCsvDataProvider(
        doc_id=core_settings.grist_data_model_doc_id,
        grist_server=core_settings.grist_server_url,
        api_key=core_settings.grist_api_key,
    )

    links_df = loader.get_table_df("Links")
    links_df = links_df[
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
    ]
    links_df = links_df.astype(str)
    links_df["has_direction"] = links_df["has_direction"].astype(bool)

    attrs_df = loader.get_table_df("Attributes")
    attrs_df = attrs_df[
        [
            "attribute_name",
            "description",
            "anchor",
            "link",
            "data_example",
            "embeddable",
            "query",
            "dtype",
            "embed_threshold",
        ]
    ]
    # attrs_df = attrs_df.astype(str)
    attrs_df["embeddable"] = attrs_df["embeddable"].astype(bool)
    attrs_df["embed_threshold"] = attrs_df["embed_threshold"].astype(float)

    anchors_df = loader.get_table_df("Anchors")
    anchors_df = anchors_df[
        [
            "noun",
            "description",
            "id_example",
            "query",
        ]
    ]
    anchors_df = anchors_df.astype(str)

    yield anchors_df, attrs_df, links_df


def parse_bool(bool_str: str) -> bool:
    return str(bool_str).lower() in ["1", "true", "да", "есть"]


def get_grist_data(batch_size: int = 500):
    """
    Fetch all anchors and links from Grist into node/edge tables
    """

    dp = GristSQLDataProvider(
        doc_id=core_settings.grist_data_doc_id,
        grist_server=core_settings.grist_server_url,
        api_key=core_settings.grist_api_key,
        batch_size=batch_size,
    )

    # Nodes
    node_records = []
    anchor_types = dp.get_anchor_types()  # does not check data model! only lists tables that are named anchor_...
    logger.info(f"Fetching {len(anchor_types)} anchor types from Grist: {anchor_types}")

    for anchor_type in anchor_types:
        try:
            anchors = dp.get_anchors(anchor_type, dm_attrs=[])  # We don't have DataModel attrs here
        except Exception as exc:
            logger.error(f"Failed to fetch anchors for type {anchor_type}: {exc}")
            continue

        for a in anchors:
            node_records.append(
                {
                    "node_id": a.id,
                    "node_type": a.type,
                    "attributes": a.data or {},
                }
            )

    nodes_df = pd.DataFrame(node_records)

    # Edges
    edge_records = []
    link_types = dp.get_link_types()
    logger.info(f"Fetching {len(link_types)} link types from Grist: {link_types}")

    for link_type in link_types:
        try:
            links = dp.get_links(link_type)
        except Exception as exc:
            logger.error(f"Failed to fetch links for type {link_type}: {exc}")
            continue

        for link in links:
            edge_records.append(
                {
                    "from_node_id": link.id_from,
                    "to_node_id": link.id_to,
                    "from_node_type": link.id_from.split(":")[0] if ":" in link.id_from else None,
                    "to_node_type": link.id_to.split(":")[0] if ":" in link.id_to else None,
                    "edge_label": link.type,
                    "attributes": link.data or {},
                }
            )

    edges_df = pd.DataFrame(edge_records)

    # preventive drop_duplicates / na records
    if not nodes_df.empty:
        nodes_df = nodes_df.dropna(subset=["node_id", "node_type"]).drop_duplicates(subset=["node_id"])
    if not edges_df.empty:
        edges_df = edges_df.dropna(subset=["from_node_id", "to_node_id", "edge_label"]).drop_duplicates(
            subset=["from_node_id", "to_node_id", "edge_label"]
        )

    yield nodes_df, edges_df


def filter_grist_nodes(df: pd.DataFrame, dm_nodes: pd.DataFrame, dm_attributes: pd.DataFrame) -> pd.DataFrame:
    """keep only those nodes that are described in data model"""

    # filter nodes
    filtered_nodes = df.loc[df.node_type.isin(dm_nodes["noun"])].copy()

    # filter attribute keys
    filtered_nodes["attributes"] = filtered_nodes["attributes"].apply(
        lambda x: {k: v for k, v in x.items() if k in dm_attributes["attribute_name"].values}
    )
    return filtered_nodes


def filter_grist_edges(df: pd.DataFrame, dm_links: pd.DataFrame) -> pd.DataFrame:
    """keep only those edges that are described in data model"""

    # add reverse links where applicable
    rev_dm_links = dm_links.loc[~dm_links.has_direction].copy()
    rev_dm_links = rev_dm_links.rename(columns={"anchor1": "anchor2", "anchor2": "anchor1"})

    dm_links = pd.concat([dm_links, rev_dm_links])
    dm_links["fr_to_code"] = dm_links["anchor1"] + "-" + dm_links["anchor1"] + "-" + dm_links["sentence"]

    df["fr_to_code"] = df["from_node_type"] + "-" + df["to_node_type"] + "-" + df["edge_label"]

    # filter edges by node types
    filtered_edges = df.loc[df.fr_to_code.isin(dm_links["fr_to_code"])].copy()

    # rm temp column
    filtered_edges = filtered_edges.drop(columns=["fr_to_code"])
    return filtered_edges


def ensure_memgraph_indexes(dm_attributes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create label / vector indices
    """

    # anchors for indices
    anchor_types: set[str] = set(dm_attributes["anchor"].dropna().unique())

    # embeddable attrs for vector indices
    vec_attr_rows = dm_attributes[(dm_attributes["embeddable"]) & (dm_attributes["dtype"].str.lower() == "str")]

    driver = GraphDatabase.driver(
        uri=core_settings.memgraph_uri,
        auth=(
            core_settings.memgraph_user,
            core_settings.memgraph_pwd,
        ),
    )

    with driver.session() as session:
        # Indices
        for label in anchor_types:
            try:
                session.run(f"CREATE INDEX ON :`{label}`(id)")
            except Exception as exc:
                logger.debug(f"CREATE INDEX failed for label {label}: {exc}")  # probably index exists

            try:
                session.run(f"CREATE CONSTRAINT ON (n:`{label}`) ASSERT n.id IS UNIQUE")
            except Exception as exc:
                logger.debug(f"CREATE CONSTRAINT failed for label {label}: {exc}")  # probably index exists

        # Vector indices
        for _, row in vec_attr_rows.iterrows():
            attr: str = row["attribute_name"]
            embeddings_dim = core_settings.embeddings_dim

            if pd.notna(row["anchor"]):
                label = row["anchor"]
            elif pd.notna(row["link"]):
                label = row["link"]  # relationship label
            else:
                continue  # cannot determine label

            idx_name = f"{label}_{attr}_embed_idx".replace(" ", "_")
            prop_name = f"{attr}_embedding"

            cypher = (
                f"CREATE VECTOR INDEX `{idx_name}` ON :`{label}`(`{prop_name}`) "
                f'WITH CONFIG {{"dimension": {embeddings_dim}, "capacity": 1024, "metric": "cos"}}'
            )
            try:
                session.run(cypher)
            except Exception as exc:
                logger.debug(f"CREATE VECTOR INDEX failed for {idx_name}: {exc}")  # probably index exists
                continue

    driver.close()

    # nominal outputs
    memgraph_indexes = pd.DataFrame({"attribute_name": list(anchor_types)})
    memgraph_vector_indexes = vec_attr_rows[["attribute_name", "anchor", "link"]].copy()
    return memgraph_indexes, memgraph_vector_indexes


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

    provider = OpenaiEmbeddingProvider(
        cache_dir=Path(core_settings.embeddings_cache_path),
        embeddings_dim=core_settings.embeddings_dim,
    )

    tasks: list[tuple[int, str, str]] = []  # (row_idx, attr_name, text)

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
        provider.close()
        return df

    texts = [t[2] for t in tasks]
    vectors = provider.get_embeddings(texts)
    provider.close()

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
        attr_dict[f"{attr_name}_embedding"] = vec.tolist()
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


if __name__ == "__main__":
    # import os
    # import dotenv
    #
    # dotenv.load_dotenv()
    # df = pd.read_sql_table("catalog_raw", con=os.environ["DB_CONN_URI"])
    # # oc, c = parse_offer_categories(df)
    # oc = pd.read_sql_table("offer_categories", con=os.environ["DB_CONN_URI"])
    # tech = pd.read_sql_table("tech_specs_names", con=os.environ["DB_CONN_URI"])
    # text = pd.read_sql_table("text_specs_names", con=os.environ["DB_CONN_URI"])
    # rels = pd.read_sql_table("related_products", con=os.environ["DB_CONN_URI"])
    # dma = next(get_anchor_attribute_map())
    # dmanchor, dmattr, dml = next(get_data_model())
    # # dml = pd.read_sql_table("dm_links", con=os.environ["DB_CONN_URI"])
    #
    # def chunk_dataframe(df: pd.DataFrame, chunk_size: int):
    #     for start in range(0, len(df), chunk_size):
    #         yield df.iloc[start : start + chunk_size]
    #
    # for i, df_chunk in enumerate(chunk_dataframe(df, chunk_size=256)):
    #     print(i)
    #     a, l = parse_catalog(df_chunk, oc, dma, dml, tech, text)
    ...
