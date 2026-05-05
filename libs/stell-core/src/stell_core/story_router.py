import dataclasses
from dataclasses import dataclass, field
from typing import Any

from stell_core.configs import ActionStep, ActiveLoopStep, IntentStep, SlotWasSetStep, StoriesConfig, StoryModel


@dataclass
class StoryRouter:
    stories: list[StoryModel] = field(default_factory=list)

    @classmethod
    def from_config(cls, config: StoriesConfig) -> "StoryRouter":
        return cls(stories=config.stories)

    def find(
        self,
        intent: str,
        entities: dict[str, Any],
        slots: dict[str, Any],
        current_story: str | None,
        current_step: int,
    ) -> StoryModel | None:
        if current_story is not None:
            story = self.get_by_name(current_story)
            if story is not None:
                expected = self.next_expected_intent(story, current_step)
                if expected is not None and expected.intent == intent:
                    return story

        candidates = [
            s for s in self.stories
            if s.steps
            and isinstance(s.steps[0], IntentStep)
            and s.steps[0].intent == intent
        ]

        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        return self.pick_most_specific(candidates, entities, slots)

    def find_for_reroute(
        self,
        intent: str,
        entities: dict[str, Any],
        slots: dict[str, Any],
        actions_done: int,
    ) -> StoryModel | None:
        candidates = [
            s for s in self.stories
            if s.steps and isinstance(s.steps[0], IntentStep) and s.steps[0].intent == intent
        ]
        matching = []
        for s in candidates:
            batch = self.next_actions(s, actions_done, slots)
            if batch is not None and batch:
                matching.append(s)
        if not matching:
            return None
        if len(matching) == 1:
            return matching[0]
        return self.pick_most_specific(matching, entities, slots)

    def next_actions(self, story: StoryModel, actions_done: int, slots: dict[str, Any]) -> list[ActionStep] | None:
        actions_seen = 0
        collecting = False
        result: list[ActionStep] = []

        for step in story.steps:
            if isinstance(step, IntentStep):
                if actions_seen == actions_done:
                    collecting = True
                elif collecting:
                    break
            elif isinstance(step, ActionStep):
                if not collecting and actions_seen == actions_done:
                    collecting = True
                if collecting:
                    result.append(step)
                actions_seen += 1
            elif isinstance(step, SlotWasSetStep):
                if collecting:
                    if result:
                        break
                    elif not slots_match(step, slots):
                        return None
                elif actions_seen == actions_done:
                    if not slots_match(step, slots):
                        return None
                    collecting = True
            elif isinstance(step, ActiveLoopStep):
                if step.active_loop is not None:
                    if collecting:
                        break
                elif not collecting and actions_seen == actions_done:
                    collecting = True
        return result

    def is_complete(self, story_name: str, actions_done: int, slots: dict[str, Any]) -> bool:
        story = self.get_by_name(story_name)
        if story is None:
            return True
        next_batch = self.next_actions(story, actions_done=actions_done, slots=slots)
        return (next_batch is not None and not next_batch) and self.next_expected_intent(story, actions_done) is None

    def get_by_name(self, name: str) -> StoryModel | None:
        return next((s for s in self.stories if s.story == name), None)

    def next_expected_intent(self, story: StoryModel, actions_done: int) -> IntentStep | None:
        actions_seen = 0
        for step in story.steps:
            if isinstance(step, IntentStep):
                if actions_seen == actions_done:
                    return step
            elif isinstance(step, ActionStep):
                actions_seen += 1
        return None

    @staticmethod
    def score(story: StoryModel, entities: dict[str, Any], slots: dict[str, Any]) -> tuple[int, int]:
        first_step = story.steps[0]
        if not isinstance(first_step, IntentStep):
            return (0, 0)

        if first_step.entities:
            satisfied = sum(
                1 for e in first_step.entities
                if e in entities or slots.get(e) is not None
            )
            entity_score = satisfied if satisfied == len(first_step.entities) else -1
        else:
            entity_score = 0

        slot_score = 0
        for step in story.steps[1:]:
            if isinstance(step, IntentStep):
                break
            if isinstance(step, SlotWasSetStep):
                slot_score = 1 if slots_match(step, slots) else -1
                break

        return (entity_score, slot_score)

    def pick_most_specific(
        self,
        candidates: list[StoryModel],
        entities: dict[str, Any],
        slots: dict[str, Any],
    ) -> StoryModel:
        return max(candidates, key=lambda c: self.score(c, entities, slots))


def slots_match(step: SlotWasSetStep, slots: dict[str, Any]) -> bool:
    for k, v in step.slot_was_set.items():
        current = slots.get(k)
        if v is dataclasses.MISSING:
            if current is None:
                return False
        elif current != v:
            return False
    return True
