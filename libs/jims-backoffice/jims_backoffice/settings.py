from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JIMS_", extra="ignore", env_file=".env")

    db_conn_uri: str

    @property
    def dsn(self) -> str:
        return self.db_conn_uri

    @property
    def async_dsn(self) -> str:
        return self.dsn.replace("postgresql://", "postgresql+asyncpg://")
