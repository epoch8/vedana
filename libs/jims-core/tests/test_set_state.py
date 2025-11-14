import pytest
import pytest_asyncio
import sqlalchemy.ext.asyncio as sa_aio
from jims_core.db import Base
from jims_core.thread.thread_context import ThreadContext
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7
from pydantic import BaseModel


@pytest_asyncio.fixture(scope="session")
async def sessionmaker() -> sa_aio.async_sessionmaker[sa_aio.AsyncSession]:
    engine = sa_aio.create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=True,
        future=True,
    )

    # Create all tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return sa_aio.async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        future=True,
    )


class TestState(BaseModel):
    some_value: str


async def pipeline_set_state(ctx: ThreadContext) -> None:
    ctx.set_state("test", TestState(some_value="test"))


@pytest.mark.asyncio
async def test_pipeline_state_write_and_load(
    sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession],
) -> None:
    thread_id = uuid7()
    contact_id = f"test:{thread_id}"
    ctl = await ThreadController.new_thread(sessionmaker, contact_id, thread_id, {})

    ctx_1 = await ctl.make_context()

    # Initially, no state
    assert ctx_1.get_state("test", TestState) is None

    await ctl.run_pipeline_with_context(pipeline_set_state, ctx_1)

    ctx_2 = await ctl.make_context()

    state = ctx_2.get_state("test", TestState)
    assert state is not None
    assert state.some_value == "test"
