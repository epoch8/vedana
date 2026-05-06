from dataclasses import dataclass
from typing import Any

from stell_core.configs import ActionStep, StoryModel
from stell_core.models import ConversationStateDTO
from stell_core.story_router import StoryRouter


@dataclass
class StoryPolicy:
    router: StoryRouter

    def predict(
        self,
        intent: str,
        entities: dict[str, Any],
        state: ConversationStateDTO,
    ) -> tuple[StoryModel, list[ActionStep]] | None:
        story = self.router.find(
            intent=intent,
            entities=entities,
            slots=state.slots,
            current_story=state.current_story,
            current_step=state.current_step,
        )
        if story is None:
            return None

        action_steps = self.router.next_actions(story, actions_done=state.current_step, slots=state.slots)
        if action_steps is None:
            return None
        return story, action_steps


@dataclass
class PolicyEnsemble:
    story_policy: StoryPolicy
    fallback_action: str = "utter_fallback"

    def next_actions(
        self,
        intent: str,
        entities: dict[str, Any],
        state: ConversationStateDTO,
    ) -> tuple[str | None, list[ActionStep]]:
        result = self.story_policy.predict(intent, entities, state)
        if result is not None:
            story, actions = result
            return story.story, actions

        return None, [ActionStep(action=self.fallback_action)]
