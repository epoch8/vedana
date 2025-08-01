import logging
import sys

import psycopg

from vedana.data_model import DataModel
from vedana.data_provider import GristSQLDataProvider
from vedana.embeddings import OpenaiEmbeddingProvider
from vedana.graph import MemgraphGraph
from vedana.importers.fast import update_graph
from vedana.settings import settings as s

logging.basicConfig(level=logging.INFO)


def main():
    data_model = DataModel.load_grist_online(
        doc_id=s.grist_data_model_doc_id, grist_server=s.grist_server_url, api_key=s.grist_api_key
    )
    print("Loaded data model:")
    print(data_model.to_text_descr())
    graph = MemgraphGraph(s.memgraph_uri, s.memgraph_user, s.memgraph_pwd, "")
    embeds = OpenaiEmbeddingProvider(s.embeddings_cache_path, s.embeddings_dim)
    with graph:
        try:
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
        except psycopg.errors.InvalidSchemaName:
            n_nodes = n_edges = 0
        print("current nodes count", n_nodes)
        print("current edges count", n_edges)

        dp = GristSQLDataProvider(s.grist_data_doc_id, grist_server=s.grist_server_url, api_key=s.grist_api_key)
        update_graph(graph, dp, data_model, embeds, node_batch_size=200, edge_batch_size=400)

        # with CsvDataProvider(s.csv_path, data_model) as dp:
        #     if input("Update graph from local files? (yes/no): ").strip().lower() != "yes":
        #         print("Exiting")
        #         return
        #
        #     print("Updating graph")
        #     update_graph_multiproc(graph, dp, data_model, embeds, node_batch_size=100, edge_batch_size=500)
        #     print("Graph updated")


if __name__ == "__main__":
    sys.exit(main())
