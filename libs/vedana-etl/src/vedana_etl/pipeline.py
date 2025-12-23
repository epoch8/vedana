from datapipe.compute import Pipeline
from datapipe.step.batch_generate import BatchGenerate
from datapipe.step.batch_transform import BatchTransform

import vedana_etl.steps as steps
from vedana_etl.catalog import (
    dm_anchor_attributes,
    dm_anchors,
    dm_conversation_lifecycle,
    dm_link_attributes,
    dm_links,
    dm_prompts,
    dm_queries,
    edges,
    grist_edges,
    grist_nodes,
    memgraph_anchor_indexes,
    memgraph_edges,
    memgraph_link_indexes,
    memgraph_nodes,
    nodes,
    rag_anchor_embeddings,
    rag_edge_embeddings,
    eval_gds,
)

data_model_steps = [
    BatchGenerate(
        func=steps.get_data_model,  # Generator with main graph data
        outputs=[
            dm_anchors,
            dm_anchor_attributes,
            dm_link_attributes,
            dm_links,
            dm_queries,
            dm_prompts,
            dm_conversation_lifecycle,
        ],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "extract"), ("stage", "data-model")],
    ),
]

grist_steps = [
    BatchGenerate(
        func=steps.get_grist_data,
        outputs=[grist_nodes, grist_edges],
        labels=[("flow", "on-demand"), ("stage", "extract"), ("stage", "grist")],
    ),
]

# ---
# This part is customisable (can be replaced with a connection of other branches

default_custom_steps = [
    BatchTransform(
        func=steps.prepare_nodes,
        inputs=[grist_nodes],
        outputs=[nodes],
        labels=[("flow", "on-demand"), ("stage", "transform"), ("stage", "grist")],
        transform_keys=["node_id"],
    ),
    BatchTransform(
        func=steps.prepare_edges,
        inputs=[grist_edges],
        outputs=[edges],
        labels=[("flow", "on-demand"), ("stage", "transform"), ("stage", "grist")],
        transform_keys=["from_node_id", "to_node_id", "edge_label"],
    ),
]

# --- Loading data to Memgraph and Vector Store ---

memgraph_steps = [
    BatchTransform(
        func=steps.ensure_memgraph_node_indexes,
        inputs=[dm_anchor_attributes],
        outputs=[memgraph_anchor_indexes],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "load")],
        transform_keys=["attribute_name"],
    ),
    BatchTransform(
        func=steps.ensure_memgraph_edge_indexes,
        inputs=[dm_link_attributes],
        outputs=[memgraph_link_indexes],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "load")],
        transform_keys=["attribute_name"],
    ),
    BatchTransform(
        func=steps.pass_df_to_memgraph,
        inputs=[nodes],
        outputs=[memgraph_nodes],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "load")],
        transform_keys=["node_id", "node_type"],
    ),
    BatchTransform(
        func=steps.pass_df_to_memgraph,
        inputs=[edges],
        outputs=[memgraph_edges],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "load")],
        transform_keys=["from_node_id", "to_node_id", "edge_label"],
    ),
    BatchTransform(
        func=steps.generate_embeddings,
        inputs=[nodes, dm_anchor_attributes],
        outputs=[rag_anchor_embeddings],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "load")],
        transform_keys=["node_id", "node_type"],
    ),
    BatchTransform(
        func=steps.generate_embeddings,
        inputs=[edges, dm_link_attributes],
        outputs=[rag_edge_embeddings],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "load")],
        transform_keys=["from_node_id", "to_node_id", "edge_label"],
    ),
]

eval_steps = [
    BatchGenerate(
        func=steps.get_eval_gds_from_grist,
        outputs=[eval_gds],
        labels=[("pipeline", "eval"), ("flow", "eval"), ("stage", "extract")],
    ),
]


def get_data_model_pipeline() -> Pipeline:
    return Pipeline(data_model_steps)


def get_pipeline(custom_steps: list) -> Pipeline:
    pipeline = Pipeline(
        [
            *data_model_steps,
            *grist_steps,
            *custom_steps,
            *memgraph_steps,
            *eval_steps,
        ]
    )

    return pipeline
