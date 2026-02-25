from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Type

import backoff
import litellm
from prometheus_client import Counter
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model: str = "gpt-4.1-nano"
    embeddings_model: str = "text-embedding-3-large"
    embeddings_dim: int = 1024

    embeddings_max_batch_size: int = 2048
    embeddings_max_tokens_per_batch: int = 200000

    # passable api_keys; if None, defaults to env vars
    model_api_key: str | None = None
    embeddings_model_api_key: str | None = None

    # openrouter_api_key: str | None = None
    openrouter_api_base_url: str = "https://openrouter.ai/api/v1"


env_settings = LLMSettings()  # type: ignore

llm_calls_total = Counter(
    "llm_calls_total",
    "Total number of LLM calls",
    ["model"],
)
llm_usage_prompt_tokens_total = Counter(
    "llm_usage_prompt_tokens_total",
    "Total number of tokens used in the prompt for LLM",
    ["model"],
)
llm_usage_completion_tokens_total = Counter(
    "llm_usage_completion_tokens_total",
    "Total number of tokens used in the completion for LLM",
    ["model"],
)


@dataclass
class ModelUsage:
    requests_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    requests_cost: float = 0

    def observe(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cached_tokens: int = 0,
        request_cost: float = 0,
    ) -> None:
        self.requests_count += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cached_tokens += cached_tokens
        self.requests_cost += request_cost


class LLMProvider:
    def __init__(self, settings: LLMSettings | None = None) -> None:
        self._settings = settings or env_settings
        self.model = self._settings.model
        self.model_api_key = self._settings.model_api_key
        self.embeddings_model = self._settings.embeddings_model
        self.embeddings_model_api_key = self._settings.embeddings_model_api_key
        self.embeddings_dim = self._settings.embeddings_dim
        self.max_batch_size = self._settings.embeddings_max_batch_size
        self.max_tokens_per_batch = self._settings.embeddings_max_tokens_per_batch

        # counters for single ThreadContext (~single pipeline run) instance
        self.usage: dict[str, ModelUsage] = defaultdict(ModelUsage)

    def set_model(self, model: str) -> None:
        self.model = model

    def observe_completion(self, completion: litellm.ModelResponse) -> None:
        usage = completion.get("usage")

        if isinstance(usage, litellm.Usage):
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            if usage.prompt_tokens_details is not None:
                cached_tokens = usage.prompt_tokens_details.cached_tokens or 0
            else:
                cached_tokens = 0
        else:
            prompt_tokens = 0
            completion_tokens = 0
            cached_tokens = 0

        self.usage[completion.model or "-"].observe(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            request_cost=completion._hidden_params.get("response_cost", 0),  # or litellm.completion_cost(completion)
        )

        llm_calls_total.labels(completion.model).inc()
        llm_usage_prompt_tokens_total.labels(completion.model).inc(prompt_tokens)
        llm_usage_completion_tokens_total.labels(completion.model).inc(completion_tokens)

    def observe_create_embedding(self, res: litellm.EmbeddingResponse) -> None:
        usage = res.usage
        model = res.model or "-"

        llm_calls_total.labels(model).inc()

        if usage is not None:
            self.usage[model].observe(
                prompt_tokens=usage.prompt_tokens,
                request_cost=res._hidden_params.get("response_cost", 0),  # or litellm.completion_cost(res)
            )
            llm_usage_prompt_tokens_total.labels(model).inc(usage.prompt_tokens)

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def create_embedding(self, text: str) -> list[float]:
        response = await litellm.aembedding(
            model=self.embeddings_model,
            input=[text],
            dimensions=self.embeddings_dim,
            api_key=self.embeddings_model_api_key,
        )
        self.observe_create_embedding(response)
        return response.data[0]["embedding"]

    def _chunk_texts(self, texts: list[str]) -> list[list[str]]:
        """Chunk texts into batches respecting both count and token limits."""
        batches: list[list[str]] = []
        batch: list[str] = []
        tokens = 0
        for text in texts:
            t = litellm.token_counter(model=self.embeddings_model, text=text)
            if t >= self.max_tokens_per_batch:
                raise ValueError(
                    f'Cannot process embedding - single text exceeds max_tokens limit '
                    f'({t} >= {self.max_tokens_per_batch}); text="{text}"'
                )
            if batch and (len(batch) >= self.max_batch_size or tokens + t >= self.max_tokens_per_batch):
                batches.append(batch)
                batch, tokens = [], 0
            batch.append(text)
            tokens += t
        if batch:
            batches.append(batch)
        return batches

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """batch method with automatic chunking"""
        results: list[list[float]] = []
        for batch in self._chunk_texts(texts):
            response = await litellm.aembedding(
                model=self.embeddings_model,
                input=batch,
                dimensions=self.embeddings_dim,
                api_key=self.embeddings_model_api_key,
            )
            self.observe_create_embedding(response)
            results.extend(d["embedding"] for d in response.data)
        return results

    def create_embedding_sync(self, text: str) -> list[float]:
        response = litellm.embedding(
            model=self.embeddings_model,
            input=[text],
            dimensions=self.embeddings_dim,
            api_key=self.embeddings_model_api_key,
        )
        self.observe_create_embedding(response)
        return response.data[0]["embedding"]

    def create_embeddings_sync(self, texts: list[str]) -> list[list[float]]:
        """batch method with automatic chunking"""
        results: list[list[float]] = []
        for batch in self._chunk_texts(texts):
            response = litellm.embedding(
                model=self.embeddings_model,
                input=batch,
                dimensions=self.embeddings_dim,
                api_key=self.embeddings_model_api_key,
            )
            self.observe_create_embedding(response)
            results.extend(d["embedding"] for d in response.data)
        return results

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def chat_completion_structured[T: BaseModel](
        self,
        messages: Iterable,  # we do not type it because litellm does not type it
        response_format: Type[T],
    ) -> T | None:
        completion = await litellm.acompletion(
            model=self.model,
            messages=list(messages),
            response_format=response_format,
            api_key=self.model_api_key,
        )
        assert isinstance(completion, litellm.ModelResponse)

        self.observe_completion(completion)

        if completion.choices:
            choice = completion.choices[0]
            assert isinstance(choice, litellm.Choices)
            content = choice.message.content
            if content:
                return response_format.model_validate_json(content)
        return None

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def chat_completion_plain(
        self,
        messages: Iterable,  # we do not type it because litellm does not type it
        use_cache: bool = False,
    ) -> litellm.Message:
        completion = await litellm.acompletion(
            model=self.model,
            messages=list(messages),
            caching=use_cache,
            api_key=self.model_api_key,
        )
        assert isinstance(completion, litellm.ModelResponse)

        self.observe_completion(completion)

        choice = completion.choices[0]
        assert isinstance(choice, litellm.Choices)

        result = choice.message

        return result

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def chat_completion_with_tools(
        self,
        messages: Iterable,  # we do not type it because litellm does not type it
        tools: list | None = None,  # we do not type it because litellm does not type it
    ) -> tuple[litellm.Message, list]:
        # todo: validate tools?
        completion = await litellm.acompletion(
            model=self.model,
            messages=list(messages),
            tools=tools,
            api_key=self.model_api_key,
        )
        assert isinstance(completion, litellm.ModelResponse)

        self.observe_completion(completion)

        choice = completion.choices[0]
        assert isinstance(choice, litellm.Choices)

        result = choice.message

        tool_calls = result.tool_calls or []
        return result, tool_calls
