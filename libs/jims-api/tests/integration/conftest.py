import pytest_asyncio
import sqlalchemy.ext.asyncio as sa_aio
from httpx import ASGITransport, AsyncClient
from jims_core.app import JimsApp
from jims_core.db import Base
from jims_core.thread.thread_context import ThreadContext

from jims_api.main import create_api


async def echo_pipeline(ctx: ThreadContext) -> None:
    last = ctx.get_last_user_message()
    ctx.send_message(f"Echo: {last}" if last else "Echo: (no message)")


@pytest_asyncio.fixture(scope="session")
async def sessionmaker() -> sa_aio.async_sessionmaker[sa_aio.AsyncSession]:
    engine = sa_aio.create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    return sa_aio.async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        future=True,
    )


@pytest_asyncio.fixture(scope="session")
async def jims_app(sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession]) -> JimsApp:
    return JimsApp(
        sessionmaker=sessionmaker,
        pipeline=echo_pipeline,
        conversation_start_pipeline=None,
    )


@pytest_asyncio.fixture
async def client(jims_app: JimsApp):
    app = create_api(jims_app, api_key=None)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
