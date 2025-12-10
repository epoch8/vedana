import logging
import traceback
from dataclasses import asdict
from typing import Any
from pydantic import BaseModel, Field

from jims_core.thread.thread_context import ThreadContext

from vedana_core.data_model import DataModel
from vedana_core.graph import Graph
from vedana_core.llm import LLM
from vedana_core.rag_agent import RagAgent
from vedana_core.settings import settings


class DataModelSelection(BaseModel):
    reasoning: str = Field(
        default="",
        description="Brief explanation of why these elements were selected for answering the user's question"
    )
    anchor_nouns: list[str] = Field(
        default_factory=list,
        description="List of anchor nouns (node types) needed to answer the question",
    )
    link_sentences: list[str] = Field(
        default_factory=list,
        description="List of link sentences (relationship types) needed to answer the question",
    )
    attribute_names: list[str] = Field(
        default_factory=list,
        description="List of attribute names needed to answer the question",
    )
    query_ids: list[str] = Field(
        default_factory=list,
        description="List of query scenario ID's that match the user's question type",
    )


dm_filter_base_system_prompt = """\
Ты — помощник по анализу структуры графовой базы данных. 

Твоя задача: проанализировать вопрос пользователя и определить, какие элементы модели данных (узлы, связи, атрибуты, сценарии запросов) необходимы для формирования ответа.

**Правила выбора:**
1. Выбирай только те элементы, которые ДЕЙСТВИТЕЛЬНО нужны для ответа на вопрос
2. Если вопрос касается связи между сущностями — выбери соответствующие узлы И связь между ними
3. Выбирай атрибуты, которые могут содержать искомую информацию или использоваться для фильтрации
4. Выбирай сценарий запроса, который лучше всего соответствует типу вопроса пользователя
5. Лучше выбрать чуть больше, чем упустить важное — но не выбирай всё подряд

**Формат ответа:**
Верни JSON с выбранными элементами. Используй ТОЧНЫЕ имена из предоставленной модели данных.
"""

dm_filter_user_prompt_template = """\
**Вопрос пользователя:**
{user_query}

**Модель данных:**
{compact_data_model}

Проанализируй вопрос и выбери необходимые элементы модели данных для формирования ответа.
"""


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
        enable_filtering: bool = True,
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
        self.enable_filtering = enable_filtering

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
        if self.enable_filtering:
            await ctx.update_agent_status("Analyzing query structure...")
            filtered_dm, filter_selection = await self.filter_data_model(query, ctx)
            self.logger.info(
                f"Data model filtered: "
                f"{len(filtered_dm.anchors)}/{len(self.data_model.anchors)} anchors, "
                f"{len(filtered_dm.links)}/{len(self.data_model.links)} links, "
                f"{len(filtered_dm.attrs)}/{len(self.data_model.attrs)} attrs, "
                f"{len(filtered_dm.queries)}/{len(self.data_model.queries)} queries"
            )
        else:
            filtered_dm = self.data_model
            filter_selection = DataModelSelection()

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
        if self.enable_filtering:
            technical_info["dm_filtering"] = {
                "filter_model": self.filter_model,
                "reasoning": filter_selection.reasoning,
                "selected_anchors": filter_selection.anchor_nouns,
                "selected_links": filter_selection.link_sentences,
                "selected_attributes": filter_selection.attribute_names,
                "selected_queries": filter_selection.query_ids,
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

    async def filter_data_model(
        self,
        query: str,
        ctx: ThreadContext,
    ) -> tuple[DataModel, DataModelSelection]:
        # Get description for filtering
        dm_json = self.data_model.to_compact_json()

        # Build the prompt
        system_prompt = self.data_model.prompt_templates().get("dm_filter_prompt", dm_filter_base_system_prompt)
        user_prompt = (
            self.data_model.prompt_templates()
            .get("dm_filter_user_prompt", dm_filter_user_prompt_template)
            .format(
                user_query=query,
                compact_data_model=dm_json,
            )
        )

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        self.logger.debug(f"Filtering data model for query: {query}")

        try:
            filter_llm = ctx.llm

            base_model = ctx.llm.model
            if self.filter_model:  # if different model specified for filtering - use it
                filter_llm.set_model(self.filter_model)

            # Use structured output to get the selection
            selection = await filter_llm.chat_completion_structured(messages, DataModelSelection)

            if base_model:  # select base model back
                ctx.llm.set_model(base_model)

            if selection is None:
                raise ValueError("LLM returned empty response")

            # parse query id's to query names - LLM often misspells arbitrary query_names
            query_names = [dm_json["queries"].get(int(i)) for i in selection.query_ids]

            self.logger.debug(
                f"Data model filter selection: "
                f"anchors={selection.anchor_nouns}, "
                f"links={selection.link_sentences}, "
                f"attrs={selection.attribute_names}, "
                f"queries={query_names}",
            )
            self.logger.debug(f"Filter reasoning: {selection.reasoning}")

            # Create filtered data model
            filtered_dm = self.data_model.filter_by_selection(
                anchor_nouns=selection.anchor_nouns,
                link_sentences=selection.link_sentences,
                attribute_names=selection.attribute_names,
                query_names=query_names,
            )

            return filtered_dm, selection

        except Exception as e:  # return full data_model
            self.logger.exception(f"Data model filtering failed: {e}. Using full data model.")
            return self.data_model, DataModelSelection(
                reasoning=f"Filtering failed: {e}. Using full data model.",
                anchor_nouns=[],
                link_sentences=[],
                attribute_names=[],
                query_names=[],
            )
