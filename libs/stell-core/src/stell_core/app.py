from jims_core.app import JimsApp

from stell_core.db import get_sessionmaker
from stell_core.dialogue_pipeline import (
    DialoguePipeline,
    action_executor,
    entity_extractor,
    form_manager,
    intent_classifier,
    policy_ensemble,
    story_router,
)


async def make_app() -> JimsApp:
    return JimsApp(
        sessionmaker=get_sessionmaker(),
        pipeline=DialoguePipeline(
            intent_classifier=intent_classifier,
            entity_extractor=entity_extractor,
            action_executor=action_executor,
            story_router=story_router,
            form_manager=form_manager,
            policy_ensemble=policy_ensemble,
        ),
    )


app = make_app()
