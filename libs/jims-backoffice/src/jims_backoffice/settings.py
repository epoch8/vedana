from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JIMS_", extra="ignore", env_file=".env")

    db_conn_uri: str
    db_use_null_pool: bool = False
    db_pool_size: int | None = None
    db_pool_max_overflow: int | None = None

    @property
    def dsn(self) -> str:
        return self.db_conn_uri

    @property
    def async_dsn(self) -> str:
        if self.dsn.startswith("postgresql://"):
            return self.dsn.replace("postgresql://", "postgresql+asyncpg://")
        elif self.dsn.startswith("sqlite://"):
            return self.dsn.replace("sqlite://", "sqlite+aiosqlite://")
        return self.dsn
