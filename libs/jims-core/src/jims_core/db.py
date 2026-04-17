import asyncio
from datetime import datetime
from functools import cache
from uuid import UUID

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as sa_pg
import sqlalchemy.ext.asyncio as sa_aio
import sqlalchemy.orm as sa_orm
import sqlalchemy.types as sa_types
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.pool import NullPool


class Base(sa_orm.DeclarativeBase):
    pass


class ThreadDB(Base):
    __tablename__ = "threads"

    thread_id: sa_orm.Mapped[UUID] = sa_orm.mapped_column(sa_types.UUID, primary_key=True)
    contact_id: sa_orm.Mapped[str] = sa_orm.mapped_column(sa_types.String, nullable=True, index=True)
    created_at: sa_orm.Mapped[datetime] = sa_orm.mapped_column(sa_types.DateTime, nullable=False)

    thread_config: sa_orm.Mapped[dict] = sa_orm.mapped_column(
        sa_pg.JSONB().with_variant(sa.JSON, "sqlite"), nullable=False
    )


class ThreadEventDB(Base):
    __tablename__ = "thread_events"

    thread_id: sa_orm.Mapped[UUID] = sa_orm.mapped_column(sa_types.UUID, primary_key=True, nullable=False)
    event_id: sa_orm.Mapped[UUID] = sa_orm.mapped_column(sa_types.UUID, primary_key=True, nullable=False)

    created_at: sa_orm.Mapped[datetime] = sa_orm.mapped_column(
        sa_types.DateTime, nullable=False, server_default=sa.func.now()
    )

    # Full event type e.g. "comm.user_message.user1"
    event_type: sa_orm.Mapped[str] = sa_orm.mapped_column(sa_types.String(255), nullable=False)

    # Domain of the event e.g. "comm"
    event_domain: sa_orm.Mapped[str] = sa_orm.mapped_column(sa_types.String(255), nullable=True)

    # Name of the event e.g. "user_message"
    event_name: sa_orm.Mapped[str] = sa_orm.mapped_column(sa_types.String(255), nullable=True)

    # The event data, stored as JSON
    event_data: sa_orm.Mapped[dict] = sa_orm.mapped_column(sa_pg.JSON().with_variant(sa.JSON, "sqlite"), nullable=True)


class DbSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JIMS_", env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
