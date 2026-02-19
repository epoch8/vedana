import asyncio
from functools import cache

import sqlalchemy as sa
import sqlalchemy.ext.asyncio as sa_aio
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.pool import NullPool


class DbSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="", env_file=".env", env_file_encoding="utf-8", extra="ignore")

    config_plane_db_conn_uri: str = "postgresql://postgres:postgres@localhost:5432"
    jims_db_conn_uri: str = "postgresql://postgres:postgres@localhost:5432"
    db_conn_uri: str = "postgresql://postgres:postgres@localhost:5432"
    db_use_null_pool: bool = False
    db_pool_size: int | None = None
    db_pool_max_overflow: int | None = None


db_settings = DbSettings()  # type: ignore


def _pool_kwargs() -> dict:
    if db_settings.db_use_null_pool:
        return {"poolclass": NullPool}
    kwargs = {}
    if db_settings.db_pool_size is not None:
        kwargs["pool_size"] = db_settings.db_pool_size
    if db_settings.db_pool_max_overflow is not None:
        kwargs["max_overflow"] = db_settings.db_pool_max_overflow
    return kwargs


@cache
def _create_async_engine(loop):
    return sa_aio.create_async_engine(
        db_settings.db_conn_uri.replace("postgresql://", "postgresql+asyncpg://").replace(
            "sqlite://", "sqlite+aiosqlite://"
        ),
        **_pool_kwargs(),
    )


def get_async_db_engine() -> sa_aio.AsyncEngine:
    return _create_async_engine(asyncio.get_event_loop())


def get_db_engine() -> sa.Engine:
    return sa.create_engine(db_settings.db_conn_uri, **_pool_kwargs())


def get_sessionmaker() -> sa_aio.async_sessionmaker[sa_aio.AsyncSession]:
    return sa_aio.async_sessionmaker(
        bind=get_async_db_engine(),
        expire_on_commit=False,
        future=True,
    )


def get_config_plane_db_engine() -> sa.Engine:
    """engine for config-plane storage, as config-plane runs on another db"""
    return sa.create_engine(db_settings.config_plane_db_conn_uri)


def get_config_plane_sessionmaker() -> sa.orm.sessionmaker:
    return sa.orm.sessionmaker(
        bind=get_config_plane_db_engine(),
        expire_on_commit=False,
        future=True,
    )
