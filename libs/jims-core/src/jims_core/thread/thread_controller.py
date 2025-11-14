import datetime
import time
from uuid import UUID

import sqlalchemy as sa
import sqlalchemy.ext.asyncio as sa_aio
from jims_core.db import ThreadDB, ThreadEventDB
from jims_core.llms.llm_provider import LLMProvider
from jims_core.schema import Pipeline
from jims_core.thread.schema import CommunicationEvent, EventEnvelope
from jims_core.thread.thread_context import ThreadContext
from jims_core.util import uuid7
from opentelemetry import trace
from prometheus_client import Counter, Histogram

tracer = trace.get_tracer("jims_core.thread_controller")


jims_pipeline_run_duration = Histogram(
    "jims_pipeline_run_duration_seconds",
    "Duration of pipeline runs in seconds",
    ["status", "pipeline"],
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120, 300, 600),
)

jims_pipeline_runs_total = Counter(
    "jims_pipeline_runs_total",
    "Total number of pipeline runs",
    ["status", "pipeline"],
)


def _get_pipeline_name(pipeline: Pipeline) -> str:
    """
    Extract the pipeline name from the Pipeline object.
    """
    if hasattr(pipeline, "__name__"):
        return pipeline.__name__  # type: ignore
    elif hasattr(pipeline, "name"):
        return pipeline.name  # type: ignore
    else:
        return str(pipeline)


class ThreadController:
    def __init__(self, sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession], thread: ThreadDB) -> None:
        self.sessionmaker = sessionmaker
        self.thread = thread

    @classmethod
    async def new_thread(
        cls,
        sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession],
        contact_id: str,
        thread_id: UUID,
        thread_config: dict,
    ) -> "ThreadController":
        """
        Create a new thread with the given ID and configuration.
        If thread with given ID already exists, return existing thread.
        """
        async with sessionmaker() as session:
            # Check if thread already exists
            stmt = sa.select(ThreadDB).filter_by(thread_id=thread_id)
            existing_thread = (await session.execute(stmt)).scalar_one_or_none()

            if existing_thread:
                return cls(sessionmaker, existing_thread)

            # Create new thread if it doesn't exist
            new_thread = ThreadDB(
                contact_id=contact_id,
                thread_id=thread_id,
                created_at=datetime.datetime.now(),
                thread_config=thread_config,
            )
            session.add(new_thread)
            await session.commit()

        ctl = cls(sessionmaker, new_thread)

        await ctl.store_event_dict(event_id=uuid7(), event_type="jims.lifecycle.thread_created", event_data={})

        return ctl

    @classmethod
    async def from_thread_id(
        cls,
        sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession],
        thread_id: UUID,
    ) -> "ThreadController | None":
        """
        Retrieve a thread by its ID.
        """
        async with sessionmaker() as session:
            stmt = sa.select(ThreadDB).filter_by(thread_id=thread_id)
            thread = (await session.execute(stmt)).scalar_one_or_none()

        if thread:
            return cls(sessionmaker, thread)
        else:
            return None

    @classmethod
    async def latest_thread_from_contact_id(
        cls,
        sessionmaker: sa_aio.async_sessionmaker[sa_aio.AsyncSession],
        from_contact_id: str,
    ) -> "ThreadController | None":
        """
        Retrieve the latest thread associated with a given contact_id.
        """
        async with sessionmaker() as session:
            stmt = (
                sa.select(ThreadDB)
                .where(ThreadDB.contact_id == from_contact_id)
                .order_by(ThreadDB.created_at.desc())
                .limit(1)
            )
            thread = (await session.execute(stmt)).scalar_one_or_none()

        if thread:
            return ThreadController(sessionmaker, thread)
        return None

    async def store_event_dict(self, event_id: UUID, event_type: str, event_data: dict) -> None:
        """
        Add an event to the thread.
        """
        async with self.sessionmaker() as session:
            new_event = ThreadEventDB(
                thread_id=self.thread.thread_id,
                event_id=event_id,
                event_type=event_type,
                event_data=event_data,
            )
            session.add(new_event)
            await session.commit()

    async def store_user_message(self, event_id: UUID, content: str) -> None:
        """
        Store a user message in the thread.
        """
        event_data = CommunicationEvent(role="user", content=content)
        await self.store_event_dict(event_id, "comm.user_message", dict(event_data))

    async def store_assistant_message(self, event_id: UUID, content: str) -> None:
        """
        Store an assistant message in the thread.
        """
        event_data = CommunicationEvent(role="assistant", content=content)
        await self.store_event_dict(event_id, "comm.assistant_message", dict(event_data))

    async def make_context(self) -> ThreadContext:
        """
        Create a new ThreadContext for the current thread.
        """
        async with self.sessionmaker() as session:
            stmt = (
                sa.select(ThreadEventDB).filter_by(thread_id=self.thread.thread_id).order_by(ThreadEventDB.created_at)
            )

            events_res = (await session.execute(stmt)).scalars().all()

        history = [
            CommunicationEvent(**event.event_data)  # type: ignore
            for event in events_res
            if event.event_type.startswith("comm.")
        ]

        events = [
            EventEnvelope(
                thread_id=self.thread.thread_id,
                event_id=event.event_id,
                created_at=event.created_at,
                event_type=event.event_type,
                event_data=event.event_data,
            )
            for event in events_res
        ]

        return ThreadContext(
            thread_id=self.thread.thread_id,
            history=history,
            events=events,
            llm=LLMProvider(),
            thread_config=self.thread.thread_config,
        )

    async def run_pipeline_with_context(
        self,
        pipeline: Pipeline,
        ctx: ThreadContext | None = None,
    ) -> list[EventEnvelope]:
        """
        Run a pipeline with the current thread context.
        """

        # Create a new ThreadContext with the current thread

        if ctx is None:
            ctx = await self.make_context()

        with tracer.start_as_current_span("jims.run_pipeline_with_context") as span:
            span.set_attribute("jims.thread.id", str(ctx.thread_id))
            span.set_attribute("jims.pipeline", _get_pipeline_name(pipeline))

            pipeline_start_time = time.time()

            # Run the pipeline with the context
            try:
                await pipeline(ctx)
            except Exception as e:
                pipeline_duration = time.time() - pipeline_start_time
                jims_pipeline_run_duration.labels(
                    status="failure",
                    pipeline=_get_pipeline_name(pipeline),
                ).observe(pipeline_duration)

                span.set_status(trace.StatusCode.ERROR, f"Pipeline execution failed: {str(e)}")
                jims_pipeline_runs_total.labels(status="failure", pipeline=_get_pipeline_name(pipeline)).inc()

                raise

            pipeline_duration = time.time() - pipeline_start_time
            jims_pipeline_run_duration.labels(
                status="success",
                pipeline=_get_pipeline_name(pipeline),
            ).observe(pipeline_duration)

            span.set_status(trace.StatusCode.OK)
            jims_pipeline_runs_total.labels(status="success", pipeline=_get_pipeline_name(pipeline)).inc()

        for event in ctx.outgoing_events:
            await self.store_event_dict(
                event_id=event.event_id,
                event_type=event.event_type,
                event_data=event.event_data,
            )

        return ctx.outgoing_events
