from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # db_conn_uri: str = "sqlite+pysqlite3:///db.sqlite"
    db_conn_uri: str = "postgresql://postgres:postgres@localhost:5432"

    grist_server_url: str  # default 'https://api.getgrist.com'
    grist_api_key: str
    grist_data_model_doc_id: str
    grist_data_doc_id: str

    embeddings_cache_path: str = "embeddings_cache.db"  # todo remove, keep embeds in datapipe only

    embeddings_dim: int = 1024

    memgraph_uri: str = "bolt://localhost:7687"  # localhost for local tests
    memgraph_user: str = "user"
    memgraph_pwd: str = "password"


settings = Settings()  # type: ignore
