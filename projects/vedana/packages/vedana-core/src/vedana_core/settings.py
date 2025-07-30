from pydantic_settings import BaseSettings, SettingsConfigDict


def get_custom_settings(prefix: str = ""):
    class CustomSettings(BaseSettings):
        model_config = SettingsConfigDict(
            env_prefix=prefix,
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
        )

        grist_server_url: str  # default 'https://api.getgrist.com'
        grist_api_key: str
        grist_data_model_doc_id: str
        grist_data_doc_id: str
        embeddings_cache_path: str = "data/cache/embeddings_cache.db"
        embeddings_dim: int = 1024
        memgraph_uri: str = "bolt://memgraph:7687"  # localhost for local tests
        memgraph_user: str = "neo4j"
        memgraph_pwd: str = "modular-current-bonjour-senior-neptune-8618"

    return CustomSettings


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    grist_server_url: str  # default 'https://api.getgrist.com'
    grist_api_key: str
    grist_data_model_doc_id: str
    grist_data_doc_id: str

    # todo remove, keep embeds in datapipe only
    embeddings_cache_path: str = "data/cache/embeddings_cache.db"
    default_embeddings_cache_path: str = "data/cache/embeddings_cache.db"
    embeddings_dim: int = 1024

    debug: bool = False
    model: str = "gpt-4.1"

    memgraph_uri: str
    memgraph_user: str
    memgraph_pwd: str


settings = Settings()  # type: ignore
