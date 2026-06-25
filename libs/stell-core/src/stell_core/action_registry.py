import random
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, TypedDict

from loguru import logger

from stell_core.constants import ACTION_LISTEN
from stell_core.models import (
    ActionExecutedEventDTO,
    BotUtterEventDTO,
    ConversationStateDTO,
    FollowupActionEventDTO,
    SlotSetEventDTO,
    TrackerEventDTO,
    UserUtteranceRevertedEventDTO,
)


def slot_set(name: str, value: Any) -> dict[str, Any]:
    return {"event": "slot", "name": name, "value": value}

def followup_action(name: str) -> dict[str, Any]:
    return {"event": "followup", "name": name}

def user_uttered(text: str, parse_data: dict[str, Any]) -> dict[str, Any]:
    return {"event": "user_uttered", "text": text, "parse_data": parse_data}

def all_slots_reset() -> dict[str, Any]:
    return {"event": "all_slot_reset"}

def user_utterance_reverted() -> dict[str, Any]:
    return {"event": "rewind"}


SlotSet = slot_set
FollowupAction = followup_action
UserUttered = user_uttered
AllSlotsReset = all_slots_reset
UserUtteranceReverted = user_utterance_reverted

class DomainDict(TypedDict):
    intents: list[str]
    entities: list[str]
    slots: dict[str, dict[str, Any]]
    responses: dict[str, list[dict[str, Any]]]
    actions: list[str]
    actions_params: dict[str, dict[str, Any]]
    forms: dict[str, Any]


@dataclass
class Tracker:
    state: ConversationStateDTO

    def get_slot(self, key: str) -> Any:
        return self.state.slots.get(key)

    @property
    def slots(self) -> dict[str, Any]:
        return self.state.slots

    @property
    def latest_message(self) -> dict[str, Any]:
        return {
            "text": self.state.latest_text,
            "intent": {"name": self.state.latest_intent},
            "entities": self.state.latest_entities,
        }

    @property
    def active_loop(self) -> dict[str, Any]:
        return {"name": self.state.active_form}

    @property
    def latest_action_name(self) -> str | None:
        return self.state.latest_action

    @property
    def events(self) -> list[dict[str, Any]]:
        return self.state.events

    @property
    def get_latest_user_event(self) -> dict[str, Any] | None:
        return self.state.latest_user_event

    @property
    def get_latest_bot_event(self) -> dict[str, Any] | None:
        return self.state.latest_bot_event

    @property
    def get_start_session_metadata(self) -> dict[str, Any]:
        return self.state.start_session_metadata
    @property
    def latest_action_params(self) -> dict[str, Any]:
        return self.state.latest_action_params


@dataclass
class CollectingDispatcher:
    messages: list[dict[str, Any]] = field(default_factory=list)

    def utter_message(
        self,
        text: str | None = None,
        image: str | None = None,
        json_message: dict[str, Any] | None  = None,
        template: str | None = None,
        response: str | None = None,
        attachment: str | None = None,
        buttons: list[dict[str, Any]] | None = None,
        elements: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        message = {
            "text": text,
            "buttons": buttons or [],
            "elements": elements or [],
            "custom": json_message or {},
            "template": response,
            "response": response,
            "image": image,
            "attachment": attachment,
        }
        message.update(kwargs)

        self.messages.append(message)


ActionCallable = Callable[[CollectingDispatcher, Tracker, DomainDict], Awaitable[list[dict]]]

class BaseAction:
    def name(self) -> str:
        raise NotImplementedError

    async def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
        **kwargs,
    ) -> list[dict]:
        raise NotImplementedError


@dataclass
class ActionRegistry:
    actions: dict[str, ActionCallable] = field(default_factory=dict)

    def register(self, name: str) -> Callable[[ActionCallable], ActionCallable]:
        def decorator(fn: ActionCallable) -> ActionCallable:
            if name in self.actions:
                logger.warning(f"Overriding existing action {name}")
            self.actions[name] = fn
            logger.info(f"Registered action {name}")
            return fn
        return decorator

    def register_action_class(self, action: BaseAction) -> None:
        if action.name() in self.actions:
            logger.warning(f"Overriding existing action {action.name()}")
        self.actions[action.name()] = action.run
        logger.info(f"Registered action class {action.name()}")

    def discover_subclasses(self) -> None:
        def collect(cls: type) -> None:
            for sub in cls.__subclasses__():
                self.register_action_class(sub())
                collect(sub)

        collect(BaseAction)

    async def run(
        self,
        action_name: str,
        state: ConversationStateDTO,
        domain: DomainDict,
        params: dict[str, Any] | None = None,
    ) -> tuple[list[dict], CollectingDispatcher]:
        fn = self.actions.get(action_name)
        if fn is None:
            raise KeyError(f"Action '{action_name}' is not registered")
        dispatcher = CollectingDispatcher()
        tracker = Tracker(state=state)
        events = await fn(dispatcher, tracker, domain, **(params or {}))
        return events, dispatcher


@dataclass
class ActionExecutor:
    registry: ActionRegistry
    domain: DomainDict

    async def execute(
        self,
        action_name: str,
        state: ConversationStateDTO,
        params: dict[str, Any] | None = None,
    ) -> list[TrackerEventDTO]:
        merged_params = {**self.domain.get("actions_params", {}).get(action_name, {}), **(params or {})}
        try:
            if action_name.startswith("utter_"):
                dispatcher = CollectingDispatcher()
                tracker = Tracker(state=state)
                await action_utter(
                    response_key=action_name,
                    dispatcher=dispatcher,
                    tracker=tracker,
                    domain=self.domain,
                    params=merged_params,
                )
                rasa_events: list[dict] = []
            else:
                rasa_events, dispatcher = await self.registry.run(
                    action_name=action_name,
                    state=state,
                    domain=self.domain,
                    params=merged_params,
                )
        except KeyError as exc:
            logger.warning(f"Action '{action_name}' failed: {exc}")
            return [ActionExecutedEventDTO(action_name=action_name, params={})]
        result: list[TrackerEventDTO] = []

        for event in rasa_events:
            event_type = event.get("event")
            if event_type == "slot":
                result.append(SlotSetEventDTO(
                    slot_name=event["name"],
                    old_value=state.slots.get(event["name"]),
                    new_value=event.get("value"),
                ))
            elif event_type == "followup":
                result.append(
                    FollowupActionEventDTO(
                        name=event["name"]
                    )
                )
            elif event_type == "rewind":
                result.append(UserUtteranceRevertedEventDTO())

        for message in dispatcher.messages:
            if message.get("text"):
                result.append(BotUtterEventDTO(
                    text=message["text"],
                    response_key=action_name if action_name.startswith("utter_") else "",
                    buttons=message.get("buttons") or [],
                ))

        result.append(ActionExecutedEventDTO(
            action_name=action_name,
            params=merged_params,
        ))

        return result


action_registry = ActionRegistry()


@action_registry.register(name=ACTION_LISTEN)
async def action_listen(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict]:
    return []


@action_registry.register(name="action_set_slot")
async def action_set_slot(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    parameter_slot_name = "slot_name"
    parameter_slot_value = "slot_value"
    slot_name = kwargs.get(parameter_slot_name)
    slot_value = kwargs.get(parameter_slot_value)
    events: list[dict[str, Any]] = []
    if slot_name is None or slot_name not in domain["slots"]:
        logger.warning(f"Parameter {slot_name} is not found!")
        return events
    if slot_value is None:
        logger.warning(f"Parameter {parameter_slot_value} is not found!")
        return events
    return [SlotSet(slot_name, slot_value)]


@action_registry.register(name="action_clear_slot")
async def action_clear_slot(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    parameter = "slot_name"
    slot_name = kwargs.get(parameter)
    if slot_name is None:
        logger.warning(f"Parameter {parameter} is not found!")
        return events
    if tracker.get_slot(slot_name) is None:
        logger.warning(f"{slot_name} is empty!")
        return events
    return [SlotSet(slot_name, None)]


@action_registry.register(name="action_copy_slot")
async def action_copy_slot(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    parameter_slot_name = "slot_name"
    parameter_copy_slot_name = "copy_slot_name"
    events: list[dict[str, Any]] = []
    slot_name = kwargs.get(parameter_slot_name)
    copy_slot_name = kwargs.get(parameter_copy_slot_name)
    if slot_name is None or slot_name not in tracker.slots:
        logger.warning(f"Parameter {slot_name} is not found!")
        return events
    if copy_slot_name is None or copy_slot_name not in tracker.slots:
        logger.warning(f"Parameter {copy_slot_name} is not found!")
        return events
    copy_slot_name_value = tracker.get_slot(copy_slot_name)
    return [SlotSet(slot_name, copy_slot_name_value)]


@action_registry.register(name="action_find_in")
async def action_find_in(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    parameter_slot_name = "slot_name"
    parameter_value = "value"
    parameter_slot_result = "slot_result_action_find_in"
    slot_name = kwargs.get(parameter_slot_name)
    value = kwargs.get(parameter_value)
    slot_result = kwargs.get(parameter_slot_result, parameter_slot_result)
    if slot_name is None or slot_name not in tracker.slots:
        logger.warning(f"Parameter {slot_name} is not found!")
        return events
    slot_value = tracker.get_slot(slot_name)
    if slot_value is None:
        logger.warning(f"{slot_name} is None")
        return events
    value = str(value)
    slot_value = str(slot_value)
    result = value.lower() in slot_value.lower()
    return [SlotSet(slot_result, result)]


@action_registry.register(name="action_delete_last_event_from_tracker")
async def action_delete_last_event_from_tracker(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    return [user_utterance_reverted()]


@action_registry.register(name="action_clear_list_slots")
async def action_clear_list_slots(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    slots = kwargs.get("slots")
    delimiter = kwargs.get("delimiter", ";")
    if slots is None:
        logger.warning("Parameter slots is None")
        return events
    for slot in slots.replace(" ", "").split(delimiter):
        if slot not in tracker.slots:
            logger.warning(f"{slot} is empty!")
            continue
        events.append(SlotSet(slot, None))
    return events


@action_registry.register(name="action_list_bool_and")
async def action_list_bool_and(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    slots = kwargs.get("slots")
    delimiter = kwargs.get("delimiter", ";")
    value = kwargs.get("value")
    result_slot = "result_list_bool_and"
    if slots is None:
        logger.warning("Parameter slots is None")
        return events
    if value is None:
        logger.warning("Parameter value is None")
    for slot in slots.replace(" ", "").split(delimiter):
        if slot not in tracker.slots:
            logger.warning(f"Slot {slot} not Found!")
            continue
        tmp = tracker.get_slot(slot)
        if tmp != value:
            return [SlotSet(result_slot, False)]
    return [SlotSet(result_slot, True)]


@action_registry.register(name="action_list_bool_or")
async def action_list_bool_or(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    slots = kwargs.get("slots")
    delimiter = kwargs.get("delimiter", ";")
    value = kwargs.get("value")
    result_slot = "result_list_bool_or"
    if slots is None:
        logger.warning("Parameter slots is None")
        return events
    if value is None:
        logger.warning("Parameter value is None")
    for slot in slots.replace(" ", "").split(delimiter):
        if slot not in tracker.slots:
            logger.warning(f"Slot {slot} not Found!")
            continue
        tmp = tracker.get_slot(slot)
        if tmp == value:
            return [SlotSet(result_slot, True)]
    return [SlotSet(result_slot, False)]


@action_registry.register(name="action_concat_slots")
async def action_concat_slots(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    slot_result = "action_concat_result"
    slot_left_name = kwargs.get("slot_name_left")
    slot_right_name = kwargs.get("slot_name_right")
    delimiter = kwargs.get("delimiter", " ")
    if slot_left_name is None or slot_right_name is None:
        logger.warning("Params slot_name_left or slot_name_right are None!")
        return events
    if slot_left_name not in tracker.slots or slot_right_name not in tracker.slots:
        logger.warning(f"Slots {slot_left_name} or {slot_right_name} are not Found in Slots!")
        return events
    slot_left = tracker.get_slot(slot_left_name)
    slot_right = tracker.get_slot(slot_right_name)
    if slot_left is None:
        slot_left = ""
    if slot_right is None:
        slot_right = ""
    if slot_result not in tracker.slots:
        logger.warning(f"Slot {slot_result} is not Found in Slots!")
        return events
    result = f"{slot_left}{delimiter}{slot_right}"
    events.append(SlotSet(slot_result, result))
    return events


@action_registry.register(name="action_compare")
async def action_compare(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    slot_name = kwargs.get("slot_name")
    result_slot = kwargs.get("slot_result_action_compare", "slot_result_action_compare")
    value = kwargs.get("value")

    if slot_name is None:
        logger.warning("action_compare: missing required param slot_name")
        return []
    if result_slot not in domain["slots"]:
        logger.warning(f"action_compare: result slot '{result_slot}' not in domain slots")
        return []

    if value == "None":
        value = None

    slot_value = tracker.get_slot(slot_name)
    result = slot_value == value
    return [SlotSet(result_slot, result)]


@action_registry.register(name="action_get_last_user_phrase")
async def action_get_last_user_phrase(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    events = []
    user_phrase_slot = "last_user_phrase"
    if user_phrase_slot not in tracker.slots:
        logger.warning(f"Slot {user_phrase_slot} not found in tracker!")
        return events
    latest_message_text = None
    for event in reversed(tracker.events):
        if event.get("event_type") == "user_input":
            latest_message_text = event.get("text")
            break
    if latest_message_text is None:
        logger.warning("Latest message text is None")
    return [SlotSet(user_phrase_slot, latest_message_text)]


@action_registry.register(name="action_reset_slots")
async def action_reset_slots(
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    **kwargs,
) -> list[dict[str, Any]]:
    slots_to_keep = []
    name: str
    not_resettable_slots = kwargs.get("not_resettable_slots", "")
    not_resettable_slots_split = not_resettable_slots.split(",")
    for name in not_resettable_slots_split:
        value = tracker.get_slot(name)
        slots_to_keep.append(SlotSet(name, value))
    return [AllSlotsReset(), *slots_to_keep]


async def action_utter(
    response_key: str,
    dispatcher: CollectingDispatcher,
    tracker: Tracker,
    domain: DomainDict,
    params: dict[str, Any] | None = None,
) -> list[dict]:
    variants = domain["responses"].get(response_key, [])
    if not variants:
        raise KeyError(f"Response '{response_key}' not found in domain")
    if params is None:
        params = {}
    variant = random.choice(variants)
    format_context = defaultdict(str, {**tracker.slots, **params})
    text = variant.get("text", "").format_map(format_context)
    buttons = variant.get("buttons", [])
    dispatcher.utter_message(text=text, buttons=buttons)
    return []
