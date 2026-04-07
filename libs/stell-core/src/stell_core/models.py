import datetime
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from jims_core.thread.schema import EventEnvelope
from jims_core.thread.thread_context import ThreadContext


class TrackerEventStatus(StrEnum):
    SAVED = "saved"
    IN_PROGRESS = "in_progress"


@dataclass(kw_only=True)
class TrackerEventDTO:
    event_id: UUID = field(default_factory=uuid4)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    event_type: str
    event_status: TrackerEventStatus = TrackerEventStatus.IN_PROGRESS

    def to_dict(self, exclude: set[str] | None = None) -> dict[str, Any]:
        exclude = exclude or set()
        return {k:v for k, v in asdict(self).items() if k not in exclude}
    
    def _generate_event_type(self) -> str:
        return f"stell.{self.event_type}"
    
    def to_event_envelope(self, thread_id: UUID) -> EventEnvelope[dict[str, Any]]:
        return EventEnvelope(
            thread_id=thread_id,
            event_id=self.event_id,
            created_at=self.created_at,
            event_type=self.event_type,
            event_data=self.to_dict(exclude={"event_id", "created_at", "event_status"})
        )
    
    
    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "TrackerEventDTO":
        raise NotImplementedError


@dataclass(kw_only=True)
class ConversationStateDTO:
    thread_id: UUID
    slots: dict[str, Any]
    active_form: str | None
    form_slot_in_progress: str | None
    current_story: str | None
    current_step: int
    latest_intent: str | None
    latest_text: str | None
    latest_entities: dict[str, Any]
    latest_action: str | None
    latest_user_event: dict[str, Any] | None
    latest_bot_event: dict[str, Any] | None
    start_session_metadata: dict[str, Any]
    events: list[dict[str, Any]]
    latest_action_params: dict[str, Any]


@dataclass(kw_only=True)
class SessionStartEventDTO(TrackerEventDTO):
    event_type: str = "session_start"

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "SessionStartEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
        )



@dataclass(kw_only=True)
class UserInputEventDTO(TrackerEventDTO):
    event_type: str = "user_input"
    text: str
    intent: str
    entities: dict[str, Any]

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "UserInputEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            text=event.event_data["text"],
            intent=event.event_data["intent"],
            entities=event.event_data["entities"],
        )


@dataclass(kw_only=True)
class BotUtterEventDTO(TrackerEventDTO):
    event_type: str = "bot_utter"
    text: str
    response_key: str  # utter_*
    buttons: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "BotUtterEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            text=event.event_data["text"],
            response_key=event.event_data["response_key"],
            buttons=event.event_data["buttons"],
        )


@dataclass(kw_only=True)
class ActionExecutedEventDTO(TrackerEventDTO):
    event_type: str = "action_executed"
    action_name: str
    params: dict[str, Any]

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "ActionExecutedEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            action_name=event.event_data["action_name"],
            params=event.event_data["params"],
        )


@dataclass(kw_only=True)
class SlotSetEventDTO(TrackerEventDTO):
    event_type: str = "slot_set"
    slot_name: str
    old_value: Any
    new_value: Any

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "SlotSetEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            slot_name=event.event_data["slot_name"],
            old_value=event.event_data["old_value"],
            new_value=event.event_data["new_value"],
        )


@dataclass(kw_only=True)
class FormActivatedEventDTO(TrackerEventDTO):
    event_type: str = "form_activated"
    form_name: str

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "FormActivatedEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            form_name=event.event_data["form_name"],
        )


class FormDeactivatedEnum(StrEnum):
    COMPLETED = "completed"
    INTERRUPTED = "interrupted"


@dataclass(kw_only=True)
class FormDeactivatedEventDTO(TrackerEventDTO):
    event_type: str = "form_deactivated"
    form_name: str
    reason: FormDeactivatedEnum

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "FormDeactivatedEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            form_name=event.event_data["form_name"],
            reason=FormDeactivatedEnum(event.event_data["reason"]),
        )


@dataclass(kw_only=True)
class FormSlotRequestedEventDTO(TrackerEventDTO):
    event_type: str = "form_slot_requested"
    form_name: str
    slot_name: str

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "FormSlotRequestedEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            form_name=event.event_data["form_name"],
            slot_name=event.event_data["slot_name"],
        )


@dataclass(kw_only=True)
class StoryStartedEventDTO(TrackerEventDTO):
    event_type: str = "story_started"
    story_name: str

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "StoryStartedEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            story_name=event.event_data["story_name"],
        )

@dataclass(kw_only=True)
class StoryCompletedEventDTO(TrackerEventDTO):
    event_type: str = "story_completed"
    story_name: str

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "StoryCompletedEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            story_name=event.event_data["story_name"],
        )


@dataclass(kw_only=True)
class StoryAbortedEventDTO(TrackerEventDTO):
    event_type: str = "story_aborted"
    story_name: str

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "StoryAbortedEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            story_name=event.event_data["story_name"],
        )


@dataclass(kw_only=True)
class FollowupActionEventDTO(TrackerEventDTO):
    event_type: str = "followup_action"
    name: str

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "FollowupActionEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            name=event.event_data["name"],
        )
    

@dataclass(kw_only=True)
class UserUtteranceRevertedEventDTO(TrackerEventDTO):
    event_type: str = "user_utterance_reverted"

    @classmethod
    def from_event_envelope(cls, event: EventEnvelope) -> "UserUtteranceRevertedEventDTO":
        return cls(
            event_id=event.event_id,
            created_at=event.created_at,
            event_status=TrackerEventStatus.SAVED,
            event_type="user_utterance_reverted",
        )


TrackerEventTypes = {i.event_type: i for i in TrackerEventDTO.__subclasses__()}


def get_tracker_event_from_event_envelope(event: EventEnvelope) -> TrackerEventDTO | None:
    tmp = event.event_data.get("event_type", "")
    if tmp in TrackerEventTypes:
        return TrackerEventTypes[tmp].from_event_envelope(event)
    return None



@dataclass(kw_only=True)
class TrackerDTO:
    thread_id: UUID
    events: list[TrackerEventDTO]
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
    updated_at: datetime.datetime = field(default_factory=datetime.datetime.now)

    @classmethod
    def from_thread_context(cls, ctx: ThreadContext) -> "TrackerDTO":
        events = []
        for event in ctx.events:
            tmp = get_tracker_event_from_event_envelope(event=event)
            if tmp is None:
                continue
            events.append(tmp)

        return cls(
            thread_id=ctx.thread_id,
            events=events,
        )


    
    def get_thread_events_envelope(self) -> list[EventEnvelope]:
        return [
            event.to_event_envelope(thread_id=self.thread_id)
            for event in self.events
            if event.event_status == TrackerEventStatus.IN_PROGRESS
        ]

    def append(self, event: TrackerEventDTO) -> None:
        self.events.append(event)
        self.updated_at = datetime.datetime.now()

    def append_batch(self, events: list[TrackerEventDTO]) -> None:
        for event in events:
            self.append(event)

    def to_state(self) -> ConversationStateDTO:
        slots: dict[str, Any] = {}
        current_story = None
        current_step = 0
        latest_intent = None
        latest_text: str | None = None
        latest_entities: dict[str, Any] = {}
        latest_action = None
        latest_action_params: dict[str, Any] = {}
        active_form = None
        form_slot_in_progress = None
        events: list[dict[str, Any]] = []
        start_session_metadata: dict[str, Any] = {}
        _pre_user_snapshot: dict[str, Any] | None = None

        for event in self.events:
            if isinstance(event, UserInputEventDTO):
                _pre_user_snapshot = {
                    "slots": dict(slots),
                    "current_story": current_story,
                    "current_step": current_step,
                    "latest_intent": latest_intent,
                    "latest_text": latest_text,
                    "latest_entities": dict(latest_entities),
                    "latest_action": latest_action,
                    "latest_action_params": dict(latest_action_params),
                    "active_form": active_form,
                    "form_slot_in_progress": form_slot_in_progress,
                    "start_session_metadata": start_session_metadata,
                    "events": list(events),
                }
                latest_intent = event.intent
                latest_text = event.text
                latest_entities = event.entities
            elif isinstance(event, UserUtteranceRevertedEventDTO):
                if _pre_user_snapshot is not None:
                    slots = _pre_user_snapshot["slots"]
                    current_story = _pre_user_snapshot["current_story"]
                    current_step = _pre_user_snapshot["current_step"]
                    latest_intent = _pre_user_snapshot["latest_intent"]
                    latest_text = _pre_user_snapshot["latest_text"]
                    latest_entities = _pre_user_snapshot["latest_entities"]
                    latest_action = _pre_user_snapshot["latest_action"]
                    latest_action_params = _pre_user_snapshot["latest_action_params"]
                    active_form = _pre_user_snapshot["active_form"]
                    form_slot_in_progress = _pre_user_snapshot["form_slot_in_progress"]
                    start_session_metadata = _pre_user_snapshot["start_session_metadata"]
                    events = _pre_user_snapshot["events"]
                    _pre_user_snapshot = None
            elif isinstance(event, SlotSetEventDTO):
                slots[event.slot_name] = event.new_value
                if event.slot_name == "start_session_metadata":
                    start_session_metadata = event.new_value
            elif isinstance(event, StoryStartedEventDTO):
                current_story = event.story_name
                current_step = 0
            elif isinstance(event, StoryCompletedEventDTO):
                current_story = None
                current_step = 0
            elif isinstance(event, ActionExecutedEventDTO):
                latest_action = event.action_name
                latest_action_params = event.params
                if current_story is not None:
                    current_step += 1
            elif isinstance(event, FormActivatedEventDTO):
                active_form = event.form_name
                form_slot_in_progress = None
            elif isinstance(event, FormSlotRequestedEventDTO):
                form_slot_in_progress = event.slot_name
            elif isinstance(event, FormDeactivatedEventDTO) and active_form == event.form_name:
                active_form = None
                form_slot_in_progress = None
            elif isinstance(event, StoryAbortedEventDTO):
                current_story = None
                current_step = 0
            events.append(event.to_dict())

        return ConversationStateDTO(
            thread_id=self.thread_id,
            slots=slots,
            current_story=current_story,
            current_step=current_step,
            latest_intent=latest_intent,
            latest_text=latest_text,
            latest_entities=latest_entities,
            latest_action=latest_action,
            active_form=active_form,
            form_slot_in_progress=form_slot_in_progress,
            events=events,
            latest_bot_event=None,
            latest_user_event=None,
            start_session_metadata=start_session_metadata,
            latest_action_params=latest_action_params,
        )


@dataclass
class TurnTrace:
    intent: str
    story_matched: str | None
    fallback_reason: str | None
    actions_executed: list[str]
    slot_changes: dict[str, Any]
    duration_ms: int


@dataclass
class DialogOutputDTO:
    message_id: str
    conversation_id: str
    bot_response: str
    trace: TurnTrace
    buttons: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
