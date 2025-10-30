import pytest
import pytest_asyncio
import sqlalchemy.ext.asyncio as sa_aio
from typing import cast
from jims_core.db import Base
from jims_core.schema import Orchestrator
from jims_core.thread.schema import PipelineState
from jims_core.thread.thread_controller import ThreadController
from jims_core.util import uuid7


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
async def test_pipeline_state_write_and_load(
    sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession],
) -> None:
    thread_id = uuid7()
    contact_id = f"test:{thread_id}"
    ctl = await ThreadController.new_thread(sessionmaker, contact_id, thread_id, {})

    class DummyOrchestrator(Orchestrator[PipelineState]):
        def route(self, ctx):
            async def pipeline(ctx):
                return None
            return pipeline

        async def orchestrate(self, ctx) -> None:
            ctx.state.current_pipeline = "alternate"  # change pipeline state

    ctx_1 = await ctl.make_context()
    initial = cast(PipelineState, ctx_1.get_or_create_pipeline_state(PipelineState))
    assert initial.current_pipeline == "main"  # default state

    await ctl.run_with_context(DummyOrchestrator())

    ctx_2 = await ctl.make_context()
    loaded = cast(PipelineState, ctx_2.get_or_create_pipeline_state(PipelineState))
    assert loaded.current_pipeline == "alternate"  # changed during run above
