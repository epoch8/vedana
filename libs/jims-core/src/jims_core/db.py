from datetime import datetime
from uuid import UUID

import sqlalchemy as sa
import sqlalchemy.dialects.postgresql as sa_pg
import sqlalchemy.orm as sa_orm
import sqlalchemy.types as sa_types


class Base(sa_orm.DeclarativeBase):
    pass


class ThreadDB(Base):
    __tablename__ = "threads"

    contact_id: sa_orm.Mapped[str] = sa_orm.mapped_column(sa_types.String, nullable=True)  # "interface:contact_id"
    thread_id: sa_orm.Mapped[UUID] = sa_orm.mapped_column(sa_types.UUID, primary_key=True)
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
