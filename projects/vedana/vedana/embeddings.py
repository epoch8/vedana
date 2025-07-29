import abc
import logging
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import openai
from sqlalchemy import create_engine, Column, String, LargeBinary, text, select
from sqlalchemy.orm import declarative_base, sessionmaker
import threading
from jims_core.llms.llm_provider import LLMProvider

logger = logging.getLogger(__name__)

Base = declarative_base()


class EmbeddingModel(Base):
    """SA model for embedding cache entries"""

    __tablename__ = "embedding_cache"
    key = Column(String, primary_key=True)
    embedding = Column(LargeBinary, nullable=False)


class EmbeddingsCache:
    def __init__(self, cache_path: str, embeddings_dim: int, db_url: Optional[str] = None):
        self.embeddings_dim = embeddings_dim
        if db_url:
            self.engine = create_engine(db_url)
        else:
            self.db_file = Path(cache_path)
            self.engine = create_engine(f"sqlite:///{self.db_file}")
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self._lock = threading.Lock()

        # Validate existing embeddings in cache
        # self._validate_cache_consistency()

    def _validate_cache_consistency(self):
        """Validate that all embeddings in cache have consistent dimensions"""
        with self.Session() as session:
            # Get all embeddings from cache
            results = session.execute(select(EmbeddingModel.key, EmbeddingModel.embedding)).all()

            if not results:
                logger.info("Empty embedding cache - no validation needed")
                return

            # Check dimensions of all cached embeddings
            embedding_lengths = []
            inconsistent_keys = []

            for key, embedding_blob in results:
                try:
                    embedding_array = np.frombuffer(embedding_blob, dtype=np.float64)
                    actual_length = len(embedding_array)
                    embedding_lengths.append(actual_length)

                    # Check if this embedding has the wrong dimension
                    if actual_length != self.embeddings_dim:
                        inconsistent_keys.append((key, actual_length))

                except Exception as e:
                    raise ValueError(f"Failed to decode embedding for key '{key}': {e}")

            # Check for dimension inconsistencies
            if inconsistent_keys:
                inconsistent_info = [f"'{key}': {length}" for key, length in inconsistent_keys[:5]]
                if len(inconsistent_keys) > 5:
                    inconsistent_info.append(f"... and {len(inconsistent_keys) - 5} more")

                error_msg = (
                    f"Embedding dimension mismatch found in cache!\n"
                    f"Expected dimension: {self.embeddings_dim}\n"
                    f"Found {len(inconsistent_keys)} embeddings with wrong dimensions:\n"
                    f"{', '.join(inconsistent_info)}\n"
                    f"This usually happens when EMBEDDINGS_DIM is changed after embeddings are cached.\n"
                    f"Solution: Clear the embedding cache or update EMBEDDINGS_DIM to match existing embeddings."
                )
                raise ValueError(error_msg)

            # Check for internal consistency (all embeddings should have same length)
            unique_lengths = set(embedding_lengths)
            if len(unique_lengths) > 1:
                length_counts = {length: embedding_lengths.count(length) for length in unique_lengths}
                error_msg = (
                    f"Inconsistent embedding dimensions found in cache!\n"
                    f"Found embeddings with different dimensions: {dict(length_counts)}\n"
                    f"Expected all embeddings to have dimension: {self.embeddings_dim}\n"
                    f"This indicates cache corruption. Solution: Clear the embedding cache."
                )
                raise ValueError(error_msg)

            # Log successful validation
            cache_dim = embedding_lengths[0] if embedding_lengths else self.embeddings_dim
            logger.info(f"Embedding cache validation successful: {len(results)} embeddings with dimension {cache_dim}")

    def get(self, text: str) -> Optional[np.ndarray]:
        with self.Session() as session:
            result = session.execute(select(EmbeddingModel.embedding).where(EmbeddingModel.key == text)).first()
            if result:
                array = np.frombuffer(result[0], dtype=np.float64)
                return array.reshape(self.embeddings_dim)
            return None

    def set(self, text: str, embedding: np.ndarray):
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        blob = embedding.tobytes()
        with self.Session() as session:
            existing = session.execute(select(EmbeddingModel).where(EmbeddingModel.key == text)).first()
            if existing:
                existing[0].embedding = blob
            else:
                session.add(EmbeddingModel(key=text, embedding=blob))
            session.commit()

    def get_many(self, texts: List[str]) -> Dict[str, np.ndarray]:
        if not texts:
            return {}
        with self.Session() as session:
            results = session.execute(select(EmbeddingModel).where(EmbeddingModel.key.in_(texts))).all()
            return {
                row[0].key: np.frombuffer(row[0].embedding, dtype=np.float64).reshape(self.embeddings_dim)
                for row in results
            }

    def set_many(self, embeddings: Dict[str, np.ndarray]):
        if not embeddings:
            return
        with self.Session() as session:
            existing_keys = set(
                row[0]
                for row in session.execute(
                    select(EmbeddingModel.key).where(EmbeddingModel.key.in_(list(embeddings.keys())))
                ).all()
            )
            for key, embedding in embeddings.items():
                if embedding.ndim > 1:
                    embedding = embedding.flatten()
                blob = embedding.tobytes()
                if key in existing_keys:
                    update_stmt = text("UPDATE embedding_cache SET embedding = :blob WHERE key = :key")
                    session.execute(update_stmt, params={"blob": blob, "key": key})
                else:
                    session.add(EmbeddingModel(key=key, embedding=blob))
            session.commit()

    def close(self):
        if hasattr(self, "engine"):
            self.engine.dispose()

    def __del__(self):
        self.close()


class EmbeddingProvider(abc.ABC):
    def __init__(self, embeddings_dim: int):
        self.embeddings_dim = embeddings_dim

    @abc.abstractmethod
    def get_embedding(self, text: str) -> np.ndarray: ...

    @abc.abstractmethod
    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        return [self.get_embedding(t) for t in texts]

    def close(self) -> None: ...


class OpenaiEmbeddingProvider(EmbeddingProvider):
    def __init__(self, cache_dir: Path, embeddings_dim: int):
        super().__init__(embeddings_dim)
        self._cache = EmbeddingsCache(cache_dir, embeddings_dim=embeddings_dim)

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding for text, using cache if available
        """
        # Check cache first
        cached_embedding = self._cache.get(text)
        if cached_embedding is not None:
            limited_text = text if len(text) <= 256 else f"{text[:256]}...(limited output)"
            logger.info(f'Hit on cached embedding for "{limited_text}"')
            return cached_embedding

        # Generate new embedding if not in cache
        with openai.OpenAI() as client:
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text,
                dimensions=self.embeddings_dim,
            )
        embedding = np.array(response.data[0].embedding)

        # Cache the new embedding
        self._cache.set(text, embedding)

        return embedding

    def split_into_batches(self, texts: List[str]):
        """Yield batches of strings where the combined character length is under a limit."""

        char_limit = 450000

        current_batch = []
        current_length = 0
        for t in texts:
            text_length = len(t)
            if current_length + text_length > char_limit:
                if current_batch:
                    logger.info(
                        f"processing batch with len {len(current_batch)} chunks and total length {sum([len(t) for t in current_batch])}"
                    )
                    yield current_batch
                current_batch = [t]
                current_length = text_length
            else:
                current_batch.append(t)
                current_length += text_length

        if current_batch:
            logger.info(
                f"processing last batch with len {len(current_batch)} chunks and total length {sum([len(t) for t in current_batch])}"
            )

            yield current_batch

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        # Try to get as many as possible from cache
        cached = self._cache.get_many(texts)
        missing = list(set([t for t in texts if t not in cached]))
        new_embeddings = {}
        if missing:

            # check chunk lengths and trim excessively long chunks
            single_text_char_limit = 17000  # openai single request token limit
            for i, t in enumerate(missing):
                if len(t) > single_text_char_limit:
                    logger.error(f"text too long for embedding:{len(t)} vs ~{single_text_char_limit} symbol limit."
                                 f"\nText header: {t[:200]}...")
                    missing[i] = missing[i][:single_text_char_limit]
                    texts[texts.index(t)] = t[:single_text_char_limit]

            for batch in self.split_into_batches(missing):
                with openai.OpenAI() as client:
                    response = client.embeddings.create(
                        model="text-embedding-3-large",
                        input=batch,
                        dimensions=self.embeddings_dim,
                    )

                for i, text in enumerate(batch):
                    emb = np.array(response.data[i].embedding)
                    new_embeddings[text] = emb
                self._cache.set_many(new_embeddings)

        # Return in the same order as input
        all_embeddings = {**cached, **new_embeddings}
        return [all_embeddings[t] for t in texts]

    def close(self) -> None:
        self._cache.close()


class LitellmEmbeddingProvider(EmbeddingProvider):
    """Embedding provider with JIMS LLMProvider (LiteLLM) backend"""

    def __init__(
        self,
        cache_dir: Path,
        embeddings_dim: int,
        llm_provider: LLMProvider | None = None,
    ):
        super().__init__(embeddings_dim)
        self._cache = EmbeddingsCache(cache_dir, embeddings_dim=embeddings_dim)
        self._llm = llm_provider or LLMProvider()

    # Single embedding
    def get_embedding(self, text: str) -> np.ndarray:

        cached_embedding = self._cache.get(text)
        if cached_embedding is not None:
            limited_text = text if len(text) <= 256 else f"{text[:256]}..."
            logger.info(f'Hit on cached embedding for "{limited_text}"')
            return cached_embedding

        emb = np.array(self._llm.create_embedding_sync(text))

        self._cache.set(text, emb)
        return emb

    # Batch embeddings
    def split_into_batches(self, texts: List[str]):
        char_limit = 450000
        current_batch: list[str] = []
        current_length = 0
        for t in texts:
            t_len = len(t)
            if current_length + t_len > char_limit:
                if current_batch:
                    logger.info(
                        f"processing batch with len {len(current_batch)} chunks and total length {sum(len(t) for t in current_batch)}"
                    )
                    yield current_batch
                current_batch = [t]
                current_length = t_len
            else:
                current_batch.append(t)
                current_length += t_len
        if current_batch:
            logger.info(
                f"processing last batch with len {len(current_batch)} chunks and total length {sum(len(t) for t in current_batch)}"
            )
            yield current_batch

    def get_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        # Get cached first
        cached = self._cache.get_many(texts)
        missing = [t for t in texts if t not in cached]
        new_embeddings: dict[str, np.ndarray] = {}

        if missing:
            single_text_char_limit = 17000  # OpenAI ~8191 tokens, todo check for other models
            for i, t in enumerate(missing):
                if len(t) > single_text_char_limit:
                    logger.error(
                        f"text too long for embedding:{len(t)} vs ~{single_text_char_limit} symbol limit.\nText header: {t[:200]}..."
                    )
                    missing[i] = t[:single_text_char_limit]
                    # Also trim original order list so indexes match
                    texts[texts.index(t)] = t[:single_text_char_limit]

            for batch in self.split_into_batches(missing):
                embs = self._llm.create_embeddings_sync(batch)
                for i, text in enumerate(batch):
                    emb_arr = np.array(embs[i])
                    new_embeddings[text] = emb_arr

                self._cache.set_many(new_embeddings)

        all_embeddings = {**cached, **new_embeddings}
        return [all_embeddings[t] for t in texts]

    def close(self) -> None:
        self._cache.close()


def main():
    from settings import settings as s

    cache = EmbeddingsCache(s.embeddings_cache_path, s.embeddings_dim)
    print(cache.get("Maytoni").shape)
    # fixed_cache = {k: (v[0] if len(v) == 1 else v) for k, v in cache.cache.items()}
    # assert fixed_cache
    # for v in fixed_cache.values():
    #     assert isinstance(v, np.ndarray)
    #     assert v.shape == (s.embeddings_dim,)
    # cache.cache = fixed_cache
    cache.close()


if __name__ == "__main__":
    main()
