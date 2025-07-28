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
        model: str = "gpt-4.1-mini"
        embeddings_model: str = "text-embedding-3-large"
        embeddings_dim: int = 1024
        embeddings_cache_path: str = "data/cache/embeddings_cache.db"
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

    debug: bool = False
    model: str = "gpt-4.1-mini"
    embeddings_model: str = "text-embedding-3-large"
    embeddings_dim: int = 1024

    embeddings_cache_path: str = "data/cache/embeddings_cache.db"
    default_embeddings_cache_path: str = "data/cache/embeddings_cache.db"

    memgraph_uri: str = "bolt://memgraph:7687"  # localhost for local tests
    memgraph_user: str = "neo4j"
    memgraph_pwd: str = "modular-current-bonjour-senior-neptune-8618"

    db_conn_uri: str = "postgresql://user:password@localhost:5432"

    telegram_bot_token: str = ""

    # graph_data_path: DirectoryPath = Path("data/Data.grist")  # local data snapshot
    # csv_path: DirectoryPath = Path("assets/")  # local data files
    # data_model_path: DirectoryPath = Path("data/Data Model.grist")  # local data model snapshot

    app_user: str = "admin"  # for Gradio UI auth
    app_pwd: str = "admin"


settings = Settings()
