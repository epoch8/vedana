import logging
import traceback
from dataclasses import asdict
from typing import Any

from jims_core.thread.thread_context import ThreadContext

from vedana_core.data_model import DataModel
from vedana_core.dm_filter import DataModelFilter, DataModelSelection
from vedana_core.graph import Graph
from vedana_core.llm import LLM
from vedana_core.rag_agent import RagAgent
from vedana_core.settings import settings


class FilteredRagPipeline:
    """RAG Pipeline with data model filtering for optimized query processing.

    This pipeline adds an initial step that filters the data model based on
    the user's query, selecting only relevant anchors, links, attributes,
    and query scenarios. This reduces token usage and improves LLM precision
    for large data models.
    """

    def __init__(
        self,
        graph: Graph,
        data_model: DataModel,
        logger: logging.Logger | None = None,
        threshold: float = 0.8,
        top_n: int = 5,
        model: str | None = None,
        filter_model: str | None = None,
    ):
        """Initialize the filtered RAG pipeline.

        Args:
            graph: The graph database connection.
            data_model: The full data model.
            logger: Logger instance.
            threshold: Similarity threshold for vector search.
            top_n: Number of top results to return.
            model: LLM model for RAG queries.
            filter_model: LLM model for data model filtering (can be cheaper/faster).
                         If None, uses the same model as RAG queries.
        """
        self.graph = graph
        self.data_model = data_model
        self.logger = logger or logging.getLogger(__name__)
        self.threshold = threshold
        self.top_n = top_n
        self.model = model or settings.model
        self.filter_model = filter_model or self.model

    async def __call__(self, ctx: ThreadContext) -> None:
        """Main pipeline execution - implements JIMS Pipeline protocol."""

        # Get the last user message
        user_query = ctx.get_last_user_message()
        if not user_query:
            ctx.send_message("I didn't receive a question. Please ask me something!")
            return

        try:
            # Update status
            await ctx.update_agent_status("Processing your question...")

            answer, agent_query_events, technical_info = await self.process_rag_query(user_query, ctx)

            # Send the answer
            ctx.send_message(answer)

            # Store technical information as an event
            ctx.send_event(
                "rag.query_processed",
                {
                    "query": user_query,
                    "answer": answer,
                    "technical_info": technical_info,
                    "threshold": self.threshold,
                },
            )

        except Exception as e:
            self.logger.exception(f"Error in RAG pipeline: {e}")
            error_msg = "An error occurred while processing the request"  # не передаем ошибку пользователю в диалог
            ctx.send_message(error_msg)

            # Store error event
            ctx.send_event(
                "rag.error",
                {
                    "query": user_query,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )

    async def process_rag_query(self, query: str, ctx: ThreadContext) -> tuple[str, list, dict[str, Any]]:
        # 1. Filter data model
        await ctx.update_agent_status("Analyzing query structure...")
        filtered_dm, filter_selection = await self._filter_data_model(query, ctx)
        self.logger.info(
            "Data model filtered: %d/%d anchors, %d/%d links, %d/%d attrs, %d/%d queries",
            len(filtered_dm.anchors),
            len(self.data_model.anchors),
            len(filtered_dm.links),
            len(self.data_model.links),
            len(filtered_dm.attrs),
            len(self.data_model.attrs),
            len(filtered_dm.queries),
            len(self.data_model.queries),
        )

        # 2. Create LLM and agent with filtered data model
        await ctx.update_agent_status("Searching knowledge base...")
        llm = LLM(ctx.llm, prompt_templates=filtered_dm.prompt_templates(), logger=self.logger)

        if self.model != llm.llm.model and settings.debug:
            llm.llm.set_model(self.model)

        agent = RagAgent(
            graph=self.graph,
            data_model=filtered_dm,
            llm=llm,
            ctx=ctx,
            logger=self.logger,
        )

        # 3. Execute RAG query (same as RagPipeline)
        (
            answer,
            agent_query_events,
            vts_queries,
            cypher_queries,
        ) = await agent.text_to_answer_with_vts_and_cypher(
            query,
            threshold=self.threshold,
            top_n=self.top_n,
        )

        technical_info: dict[str, Any] = {
            "vts_queries": [str(q) for q in vts_queries],
            "cypher_queries": [str(q) for q in cypher_queries],
            "num_vts_queries": len(vts_queries),
            "num_cypher_queries": len(cypher_queries),
            "model_used": self.model,
            "model_stats": {m: asdict(u) for m, u in ctx.llm.usage.items()},
        }

        # Add filtering info
        technical_info["dm_filtering"] = {
            "enabled": True,
            "filter_model": self.filter_model,
            "reasoning": filter_selection.reasoning,
            "selected_anchors": filter_selection.anchor_nouns,
            "selected_links": filter_selection.link_sentences,
            "selected_attributes": filter_selection.attribute_names,
            "selected_queries": filter_selection.query_names,
            "original_counts": {
                "anchors": len(self.data_model.anchors),
                "links": len(self.data_model.links),
                "attrs": len(self.data_model.attrs),
                "queries": len(self.data_model.queries),
            },
            "filtered_counts": {
                "anchors": len(filtered_dm.anchors),
                "links": len(filtered_dm.links),
                "attrs": len(filtered_dm.attrs),
                "queries": len(filtered_dm.queries),
            },
        }

        return answer, agent_query_events, technical_info

    async def _filter_data_model(
        self, query: str, ctx: ThreadContext
    ) -> tuple[DataModel, DataModelSelection]:
        """Filter the data model based on user query."""

        # Create a separate LLM provider for filtering (can use cheaper model)
        from jims_core.llms.llm_provider import LLMProvider

        filter_llm = LLMProvider()
        if self.filter_model:
            filter_llm.set_model(self.filter_model)

        dm_filter = DataModelFilter(
            llm_provider=filter_llm,
            logger=self.logger,
        )

        return await dm_filter.filter_data_model(self.data_model, query)
