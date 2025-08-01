from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Datapipe connection URI
    # db_conn_uri: str = "sqlite+pysqlite3:///db.sqlite"
    db_conn_uri: str


settings = Settings()  # type: ignore
