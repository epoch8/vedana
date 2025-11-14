from dataclasses import dataclass
from uuid import UUID

import sqlalchemy.ext.asyncio as sa_aio

from jims_core.schema import Pipeline
from jims_core.thread.thread_controller import ThreadController


@dataclass
class JimsApp:
    sessionmaker: sa_aio.async_sessionmaker
    pipeline: Pipeline
    conversation_start_pipeline: Pipeline | None = None

    async def new_thread(self, contact_id: str, thread_id: UUID, thread_config: dict) -> ThreadController:
        return await ThreadController.new_thread(self.sessionmaker, contact_id, thread_id, thread_config)
