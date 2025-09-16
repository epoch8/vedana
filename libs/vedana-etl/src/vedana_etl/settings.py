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

    # Tests pipeline (vedana-eval) settings.
    grist_test_set_doc_id: str = ""  # optional. If not provided - the pipeline is build without tests branch
    gds_table_name: str = "Gds"  # Table names in the test set doc
    tests_table_name: str = "Tests_auto"
    judge_model: str = "gpt-5-mini"
    test_environment: str = "local"


settings = Settings()  # type: ignore
