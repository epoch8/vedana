from datapipe.compute import Pipeline
from datapipe.step.batch_generate import BatchGenerate
from datapipe.step.batch_transform import BatchTransform

import vedana_etl.steps as steps
from vedana_etl.catalog import (
    dm_anchors,
    dm_attributes,
    dm_links,
    edges,
    grist_edges,
    grist_edges_filtered,
    grist_nodes,
    grist_nodes_filtered,
    memgraph_edges,
    memgraph_indexes,
    memgraph_nodes,
    memgraph_vector_indexes,
    nodes,
)

data_model_steps = [
    BatchGenerate(
        func=steps.get_data_model,  # Generator with main graph data
        outputs=[dm_anchors, dm_attributes, dm_links],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "extract"), ("stage", "data-model")],
    ),
]

grist_steps = [
    BatchGenerate(
        func=steps.get_grist_data,
        outputs=[grist_nodes, grist_edges],
        labels=[("flow", "on-demand"), ("stage", "extract"), ("stage", "grist")],
    ),
    BatchTransform(
        func=steps.filter_grist_nodes,
        inputs=[grist_nodes, dm_anchors, dm_attributes],
        outputs=[grist_nodes_filtered],
        labels=[("flow", "on-demand"), ("stage", "transform"), ("stage", "grist")],
        transform_keys=["node_id"],
    ),
    BatchTransform(
        func=steps.filter_grist_edges,
        inputs=[grist_edges, dm_links],
        outputs=[grist_edges_filtered],
        labels=[("flow", "on-demand"), ("stage", "transform"), ("stage", "grist")],
        transform_keys=["from_node_id", "to_node_id", "edge_label"],
    ),
]

# ---
# This part is customisable (can be replaced with a connection of other branches

default_custom_steps = [
    BatchTransform(
        func=steps.prepare_nodes,
        inputs=[grist_nodes_filtered],
        outputs=[nodes],
        labels=[("flow", "on-demand"), ("stage", "transform"), ("stage", "grist")],
        transform_keys=["node_id"],
    ),
    BatchTransform(
        func=steps.prepare_edges,
        inputs=[grist_edges_filtered],
        outputs=[edges],
        labels=[("flow", "on-demand"), ("stage", "transform"), ("stage", "grist")],
        transform_keys=["from_node_id", "to_node_id", "edge_label"],
    ),
]

# ---

memgraph_steps = [
    BatchTransform(
        func=steps.ensure_memgraph_indexes,
        inputs=[dm_attributes],
        outputs=[memgraph_indexes, memgraph_vector_indexes],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "load")],
        transform_keys=["attribute_name"],
    ),
    # TODO move embeddings to pgvector, store embeddings persistently
    # Add embeddings and upload result to memgraph.
    # generate_embeddings is a last processing step, making DataFrame ready for upload
    BatchTransform(
        func=steps.generate_embeddings,
        inputs=[nodes, memgraph_vector_indexes],
        outputs=[memgraph_nodes],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "load")],
        transform_keys=["node_id", "node_type"],
        chunk_size=100,
    ),
    BatchTransform(
        func=steps.generate_embeddings,
        inputs=[edges, memgraph_vector_indexes],
        outputs=[memgraph_edges],
        labels=[("flow", "regular"), ("flow", "on-demand"), ("stage", "load")],
        transform_keys=["from_node_id", "to_node_id", "edge_label"],
        chunk_size=300,
    ),
]


# Automated testing
eval_steps = [
    BatchGenerate(
        func=steps.get_data_model_snapshot,
        outputs=["dm_version"],
        labels=[("flow", "eval"), ("stage", "extract")],
    ),
    BatchGenerate(
        func=steps.get_eval_gds_from_grist,
        outputs=["eval_gds"],
        labels=[("flow", "eval"), ("stage", "extract")],
    ),
]


def get_pipeline(custom_steps: list):
    pipeline = Pipeline(
        [
            *data_model_steps,
            *grist_steps,
            *custom_steps,
            *memgraph_steps,
        ]
    )

    return pipeline
