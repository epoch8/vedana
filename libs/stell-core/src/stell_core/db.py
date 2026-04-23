import sqlalchemy as sa
import sqlalchemy.ext.asyncio as sa_aio
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.pool import NullPool


class DbSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JIMS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_conn_uri: str = "postgresql://postgres:postgres@localhost:5432"
    db_use_null_pool: bool = False


db_settings = DbSettings()  # type: ignore


def get_db_engine() -> sa.Engine:
    kwargs = {"poolclass": NullPool} if db_settings.db_use_null_pool else {}
    return sa.create_engine(db_settings.db_conn_uri, **kwargs)


def get_async_db_engine() -> sa_aio.AsyncEngine:
    url = db_settings.db_conn_uri.replace("postgresql://", "postgresql+asyncpg://").replace(
        "sqlite://", "sqlite+aiosqlite://"
    )
    kwargs = {"poolclass": NullPool} if db_settings.db_use_null_pool else {}
    return sa_aio.create_async_engine(url, **kwargs)


def get_sessionmaker() -> sa_aio.async_sessionmaker[sa_aio.AsyncSession]:
    return sa_aio.async_sessionmaker(
        bind=get_async_db_engine(),
        expire_on_commit=False,
        future=True,
    )
