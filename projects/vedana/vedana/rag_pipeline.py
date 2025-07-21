import logging
import traceback
from dataclasses import asdict
from typing import Any

from jims_core.thread.thread_context import ThreadContext

from vedana.data_model import DataModel
from vedana.embeddings import EmbeddingProvider
from vedana.graph import Graph
from vedana.llm import LLM
from vedana.rag_agent import RagAgent
from vedana.settings import settings


class StartPipeline:
    """
    Response for /start command
    """

    def __init__(self, data_model: DataModel) -> None:
        self.lifecycle_events = data_model.conversation_lifecycle_events()
        self.start_response = self.lifecycle_events.get("/start")

    async def __call__(self, ctx: ThreadContext) -> None:
        if not self.start_response:
            ctx.send_message("Bot online. No response for /start command in LifecycleEvents")
            return

        # await ctx.update_agent_status("thinking")

        ctx.send_message(self.start_response)


class RagPipeline:
    """RAG Pipeline that implements JIMS Pipeline protocol for conversation context"""

    def __init__(
        self,
        graph: Graph,
        embed_provider: EmbeddingProvider,
        data_model: DataModel,
        logger,
        threshold: float = 0.8,
        top_n: int = 5,
        temperature: float = 0.0,
        model: str | None = None,
    ):
        self.graph = graph
        self.embed_provider = embed_provider
        self.data_model = data_model
        self.logger = logger or logging.getLogger(__name__)
        self.threshold = threshold
        self.top_n = top_n
        self.temperature = temperature
        self.model = model or settings.model

    async def __call__(self, ctx: ThreadContext) -> None:
        """Main pipeline execution - implements JIMS Pipeline protocol"""

        # Get the last user message
        user_query = ctx.get_last_user_message()
        if not user_query:
            ctx.send_message("I didn't receive a question. Please ask me something!")
            return

        try:
            # Update status
            await ctx.update_agent_status("Processing your question...")

            # Process the query using RAG
            answer, technical_info = await self._process_rag_query(user_query, ctx)

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
                    "temperature": self.temperature,
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

    async def _process_rag_query(self, query: str, ctx: ThreadContext) -> tuple[str, dict[str, Any]]:
        """Process a RAG query and return human answer and technical info"""

        llm = LLM(ctx.llm, logger=self.logger)

        if self.model != llm.llm.model and settings.debug:  # settings.model = env
            llm.llm.set_model(self.model)

        agent = RagAgent(self.graph, self.embed_provider, self.data_model, llm, self.logger, ctx=ctx)

        # Use the most comprehensive RAG method that provides human-readable answers
        (
            answer,
            vts_queries,
            cypher_queries,
        ) = await agent.text_to_answer_with_vts_and_cypher(
            query,
            threshold=self.threshold,
            temperature=self.temperature,
            top_n=self.top_n,
        )

        # Prepare technical information
        technical_info = {
            "vts_queries": [str(q) for q in vts_queries],
            "cypher_queries": [str(q) for q in cypher_queries],
            "num_vts_queries": len(vts_queries),
            "num_cypher_queries": len(cypher_queries),
            "model_used": self.model,
            "model_stats": {m: asdict(u) for m, u in ctx.llm.usage.items()},
        }

        return answer, technical_info
