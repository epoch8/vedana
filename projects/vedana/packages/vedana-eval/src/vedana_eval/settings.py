from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Grist test set doc id
    grist_test_set_doc_id: str

    # Table names in the test set doc
    gds_table_name: str = "Gds"
    tests_table_name: str = "Tests_auto"

    # LLM judge
    judge_model: str = "gpt-5-mini"

    # Misc meta
    test_environment: str = "local"


settings = Settings()  # type: ignore
