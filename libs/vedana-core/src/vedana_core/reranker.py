from typing import Protocol

from jims_core.llms.llm_provider import LLMProvider, RerankResultItem


__all__ = ["RerankResultItem", "Reranker", "CohereReranker"]


class Reranker(Protocol):
    async def rerank(
        self,
        llm_provider: "LLMProvider",
        query: str,
        docs: list[str],
        top_n: int,
    ) -> list[RerankResultItem]: ...


class CohereReranker:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.api_key = api_key

    async def rerank(
        self,
        llm_provider: LLMProvider,
        query: str,
        docs: list[str],
        top_n: int,
    ) -> list[RerankResultItem]:
        return await llm_provider.arerank(query, docs, top_n, model=self.model, api_key=self.api_key)
