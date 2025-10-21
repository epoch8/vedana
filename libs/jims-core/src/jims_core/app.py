from dataclasses import dataclass
from uuid import UUID

import sqlalchemy.ext.asyncio as sa_aio

from jims_core.schema import Orchestrator
from jims_core.thread.thread_controller import ThreadController


@dataclass
class JimsApp:
    sessionmaker: sa_aio.async_sessionmaker
    orchestrator: Orchestrator

    async def new_thread(self, thread_id: UUID, thread_config: dict) -> ThreadController:
        return await ThreadController.new_thread(self.sessionmaker, thread_id, thread_config)
