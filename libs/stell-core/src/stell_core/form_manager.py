from dataclasses import dataclass
from typing import Any

from loguru import logger

from stell_core.action_registry import CollectingDispatcher, DomainDict, Tracker, action_utter
from stell_core.models import (
    BotUtterEventDTO,
    ConversationStateDTO,
    FormDeactivatedEnum,
    FormDeactivatedEventDTO,
    FormSlotRequestedEventDTO,
    TrackerDTO,
)


@dataclass
class FormManager:
    domain: DomainDict

    def required_slots(self, form_name: str) -> list[str]:
        form_config = self.domain["forms"].get(form_name, {})
        return form_config.get("required_slots", [])

    def next_missing_slot(self, form_name: str, slots: dict[str, Any]) -> str | None:
        for slot_name in self.required_slots(form_name):
            if slots.get(slot_name) is None:
                return slot_name
        return None

    async def handle_turn(
        self,
        form_name: str,
        state: ConversationStateDTO,
        tracker: TrackerDTO,
    ) -> bool:
        missing_slot = self.next_missing_slot(form_name, state.slots)

        if missing_slot is None:
            tracker.append(FormDeactivatedEventDTO(form_name=form_name, reason=FormDeactivatedEnum.COMPLETED))
            logger.info(f"Form '{form_name}' completed - all required slots filled")
            return False

        logger.info(f"Form '{form_name}': requesting slot '{missing_slot}'")
        tracker.append(FormSlotRequestedEventDTO(form_name=form_name, slot_name=missing_slot))

        response_key = f"utter_ask_{missing_slot}"
        dispatcher = CollectingDispatcher()
        tracker_obj = Tracker(state=state)
        try:
            await action_utter(response_key, dispatcher, tracker_obj, self.domain, {})
            tracker.append_batch([
                BotUtterEventDTO(
                    text=msg["text"],
                    response_key=response_key,
                    buttons=msg.get("buttons") or [],
                ) for msg in dispatcher.messages if msg.get("text")
            ])
        except KeyError:
            logger.warning(f"Form '{form_name}': response '{response_key}' not found in domain")
        return True
