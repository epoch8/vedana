from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()


class VedanaCoreSettings(BaseSettings):
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
    model: str = "gpt-4.1"
    embeddings_model: str = "text-embedding-3-large"
    embeddings_dim: int = 1024

    pipeline_history_length: int = 20

    memgraph_uri: str
    memgraph_user: str
    memgraph_pwd: str


settings = VedanaCoreSettings()  # type: ignore
