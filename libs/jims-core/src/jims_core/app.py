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
    
    async def new_thread_via_external_id(self, external_id: str, thread_config: dict, contact_id: str | None = None) -> ThreadController:
        return await ThreadController.new_thread_via_external_id(
            sessionmaker=self.sessionmaker, external_id=external_id, contact_id=contact_id, thread_config=thread_config, 
        )
