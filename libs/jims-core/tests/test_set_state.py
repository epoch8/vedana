import pytest
import pytest_asyncio
import sqlalchemy.ext.asyncio as sa_aio
from jims_core.db import Base
from jims_core.thread.thread_context import ThreadContext
from jims_core.thread.thread_controller import ThreadController
from jims_core.thread.schema import PipelineState
from jims_core.schema import Orchestrator
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


@pytest.mark.asyncio
async def test_set_state(sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession]) -> None:
    thread_id = uuid7()
    contact_id = f"test:{thread_id}"
    ctl = await ThreadController.new_thread(sessionmaker, contact_id, thread_id, {})

    class State(BaseModel):
        key: str

    class FuncOrchestrator(Orchestrator[PipelineState]):
        def __init__(self, func):
            self.state = PipelineState()
            self._func = func

        def route(self, ctx):
            return self._func

        async def orchestrate(self, ctx):
            await self._func(ctx)

    async def expect_no_state(ctx: ThreadContext) -> None:
        state = ctx.get_state(State)
        assert state is None, "Expected no state to be set"

    await ctl.run_with_context(FuncOrchestrator(expect_no_state))

    async def set_state(ctx: ThreadContext) -> None:
        ctx.set_state(State(key="value"))

    await ctl.run_with_context(FuncOrchestrator(set_state))

    async def get_state(ctx: ThreadContext) -> None:
        state = ctx.get_state(State)
        assert state == State(key="value")

    await ctl.run_with_context(FuncOrchestrator(get_state))
