import asyncio
from functools import cache

import sqlalchemy as sa
import sqlalchemy.ext.asyncio as sa_aio
from pydantic_settings import BaseSettings, SettingsConfigDict


class DbSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JIMS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    db_conn_uri: str = "postgresql://postgres:postgres@localhost:5432"


db_settings = DbSettings()  # type: ignore


# This is needed because each async loop needs its own engine
@cache
def _create_async_engine(loop):
    return sa_aio.create_async_engine(
        db_settings.db_conn_uri.replace("postgresql://", "postgresql+asyncpg://").replace(
            "sqlite://", "sqlite+aiosqlite://"
        )
    )


def get_async_db_engine() -> sa_aio.AsyncEngine:
    return _create_async_engine(asyncio.get_event_loop())


def get_db_engine() -> sa.Engine:
    return sa.create_engine(db_settings.db_conn_uri)


def get_sessionmaker() -> sa_aio.async_sessionmaker[sa_aio.AsyncSession]:
    return sa_aio.async_sessionmaker(
        bind=get_async_db_engine(),
        expire_on_commit=False,
        future=True,
    )
