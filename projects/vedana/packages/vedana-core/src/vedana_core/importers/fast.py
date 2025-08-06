import logging
from vedana_core.data_model import DataModel
from vedana_core.graph import Graph

logger = logging.getLogger(__name__)


class DataModelLoader:
    # Todo check DataModelLoader in DataModel and move this there
    def __init__(self, data_model: DataModel, graph: Graph) -> None:
        self.data_model = data_model
        self.graph = graph

    def update_data_model_node(self):
        """Persist current DataModel inside the target graph as a dedicated node.
        A single node with label `DataModel` and fixed id is used. The whole
        model is stored as JSON in the `content` property so that application
        instances can recreate the object without talking to Grist.
        """
        try:
            self.graph.run_cypher(
                "MERGE (dm:DataModel {id: 'data_model'}) SET dm.content = $content, dm.updated_at = datetime()",
                {"content": self.data_model.to_json()},
            )
            logger.info("DataModel node updated in graph")
        except Exception as exc:
            logger.error("Failed to update DataModel node: %s", exc)
