import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from uuid import UUID

from jims_core.thread.thread_context import ThreadContext
from loguru import logger

import stell_core.actions  # noqa: F401
from stell_core.action_registry import ActionExecutor, DomainDict, action_registry
from stell_core.configs import ActionStep, yaml_config
from stell_core.constants import ACTION_LISTEN
from stell_core.form_manager import FormManager
from stell_core.models import (
    ActionExecutedEventDTO,
    BotUtterEventDTO,
    FormActivatedEventDTO,
    SlotSetEventDTO,
    StoryAbortedEventDTO,
    StoryCompletedEventDTO,
    StoryStartedEventDTO,
    TrackerDTO,
    TurnTrace,
    UserInputEventDTO,
)
from stell_core.nlu import IntentClassifier, LLMEnititiesExtractor, NLUResult, build_nlu_history, nlu_pipeline_runner
from stell_core.policy import PolicyEnsemble, StoryPolicy
from stell_core.story_router import StoryRouter

dialog_lock: set[UUID] = set()


domain = cast(DomainDict, yaml_config.domain)

intent_classifier = IntentClassifier(intents_config=yaml_config.intents, model=yaml_config.effective_llm_model)
entity_extractor = LLMEnititiesExtractor(entities=domain["entities"], model=yaml_config.effective_llm_model)
action_executor = ActionExecutor(registry=action_registry, domain=domain)

action_registry.discover_subclasses()

logger.debug(
    f"NLU config: provider={yaml_config.llm_provider!r} model={yaml_config.effective_llm_model!r} "
    f"api={yaml_config.openai_base_url or yaml_config.ollama_base_url!r} "
    f"intents={len(yaml_config.intents.intents)} entities={len(domain['entities'])}"
)

story_router = StoryRouter.from_config(yaml_config.stories)
policy_ensemble = PolicyEnsemble(story_policy=StoryPolicy(router=story_router))
form_manager = FormManager(domain=domain)


@asynccontextmanager
async def dialog_lock_contextmanager(conversation_id: UUID) -> AsyncIterator[None]:
    if conversation_id in dialog_lock:
        raise RuntimeError(f"Conversation '{conversation_id}' is already being processed")
    dialog_lock.add(conversation_id)
    try:
        yield
    finally:
        dialog_lock.discard(conversation_id)



class DialoguePipeline:
    def __init__(
        self,
        intent_classifier: IntentClassifier,
        entity_extractor: LLMEnititiesExtractor,
        action_executor: ActionExecutor,
        story_router: StoryRouter,
        form_manager: FormManager,
        policy_ensemble: PolicyEnsemble,
    ):
        self.intent_classifier = intent_classifier
        self.entity_extractor = entity_extractor
        self.action_executor = action_executor
        self.story_router = story_router
        self.form_manager = form_manager
        self.policy_ensemble = policy_ensemble

    async def __call__(self, ctx: ThreadContext) -> None:
        async with dialog_lock_contextmanager(ctx.thread_id):
            t0 = time.perf_counter()
            user_input = ctx.get_last_user_message()
            
            if user_input is None:
                return None
            
            tracker = TrackerDTO.from_thread_context(ctx=ctx)

            history = build_nlu_history(tracker.events) or None
            pre_state = tracker.to_state()
            active_slot = pre_state.form_slot_in_progress if pre_state.active_form else None

            if user_input.startswith("/"): 
                raw = user_input[1:]
                intent_str = raw.strip()
                entities: dict[str, Any] = {}

                if "{" in raw:
                    intent_part, json_part = raw.split("{", 1)
                    intent_str = intent_part.strip()
                    try:
                        entities = json.loads("{" + json_part)
                    except json.JSONDecodeError:
                        entities = {}

                nlu_result = NLUResult(
                    intent=intent_str,
                    entities=entities,
                    user_input_text=user_input,
                )

            else:
                nlu_result = await nlu_pipeline_runner(
                    user_input,
                    entity_extractor=self.entity_extractor,
                    intent_classifier=self.intent_classifier,
                    history=history,
                    active_slot=active_slot
                )
            
            logger.info(f"NLU: intent={nlu_result.intent!r} entities={nlu_result.entities}")

            tracker.append(
                UserInputEventDTO(
                    text=user_input,
                    intent=nlu_result.intent,
                    entities=nlu_result.entities,
                )
            )
            original_intent = nlu_result.intent
            story_name: str | None = None
            actions: list[ActionStep] = []
            fallback_reason: str | None = None
            known_slots = domain["slots"]
            state = tracker.to_state()
            slots_before = dict(state.slots)
            actions_executed: list[str] = []
            requested_slot = state.form_slot_in_progress if state.active_form else None

            tracker.append_batch([
                SlotSetEventDTO(
                    slot_name=e_n,
                    old_value=state.slots.get(e_n),
                    new_value=e_v,
                ) for e_n, e_v in nlu_result.entities.items() if e_n in known_slots
            ])

            state = tracker.to_state()

            if state.active_form is not None and requested_slot and state.slots.get(requested_slot) is None:
                tracker.append(SlotSetEventDTO(slot_name=requested_slot, old_value=None, new_value=user_input))
                state = tracker.to_state()

            turn_start_idx = len(tracker.events)
            actions_done = state.current_step

            if state.active_form is not None:
                form_still_collecting = await self.form_manager.handle_turn(
                    form_name=state.active_form,
                    state=state,
                    tracker=tracker,
                )
                state = tracker.to_state()
                if not form_still_collecting:
                    actions_done = state.current_step
                story_name = state.current_story

            if state.active_form is None:
                if story_name is None:
                    story_name, actions = self.policy_ensemble.next_actions(
                        intent=nlu_result.intent,
                        entities=nlu_result.entities,
                        state=state,
                    )
                    logger.info(f"Policy: story={story_name!r}")
                    fallback_reason = f"no story matched for intent '{nlu_result.intent}'" if story_name is None else None
                    if story_name is not None and state.current_story != story_name:
                        tracker.append(StoryStartedEventDTO(story_name=story_name))
                        state = tracker.to_state()

                for _ in range(50):
                    if story_name is None:
                        for step in actions:
                            if step.action == ACTION_LISTEN:
                                break
                            events = await self.action_executor.execute(action_name=step.action, state=state, params=step.params)
                            actions_executed.append(step.action)
                            tracker.append_batch(events=events)
                            state = tracker.to_state()
                            actions_done += 1
                        break
                    story_model = self.story_router.get_by_name(story_name)
                    if story_model is None:
                        break

                    batch = self.story_router.next_actions(story_model, actions_done, state.slots)

                    if batch is None:
                        new_story = self.story_router.find_for_reroute(original_intent, nlu_result.entities, state.slots, actions_done)
                        if new_story is None:
                            logger.warning(
                                f"No branch matched after re-route: intent={original_intent!r} actions_done={actions_done}"
                            )
                            story_name = None
                            break
                        story_name = new_story.story
                        if state.current_story != story_name:
                            tracker.append(StoryStartedEventDTO(story_name=story_name))
                            state = tracker.to_state()
                        logger.info(f"Re-routed to story={story_name!r} at actions_done={actions_done}")
                        continue

                    if not batch:
                        break

                    hit_listen = False
                    for step in batch:
                        if step.action == ACTION_LISTEN:
                            hit_listen = True
                            break
                        if step.action in domain["forms"]:
                            tracker.append(FormActivatedEventDTO(form_name=step.action))
                            tracker.append(ActionExecutedEventDTO(action_name=step.action, params={}))
                            actions_executed.append(step.action)
                            state = tracker.to_state()
                            actions_done += 1
                            form_still_collecting = await self.form_manager.handle_turn(
                                form_name=step.action,
                                state=state,
                                tracker=tracker,
                            )
                            state = tracker.to_state()
                            if form_still_collecting:
                                hit_listen = True
                            break
                        events = await self.action_executor.execute(action_name=step.action, state=state, params=step.params)
                        actions_executed.append(step.action)
                        tracker.append_batch(events=events)
                        state = tracker.to_state()
                        actions_done += 1

                    if (story_model is not None and self.story_router.next_expected_intent(story_model, actions_done) is not None) or hit_listen:
                        break

            if story_name is None and state.current_story is not None:
                tracker.append(StoryAbortedEventDTO(story_name=state.current_story))
                state = tracker.to_state()

            story_done = (
                state.active_form is None
                and story_name is not None
                and self.story_router.is_complete(story_name, actions_done, state.slots)
            )
            if story_done and story_name is not None:
                tracker.append(StoryCompletedEventDTO(story_name=story_name))
                state = tracker.to_state()

            bot_utters = [e for e in tracker.events[turn_start_idx:] if isinstance(e, BotUtterEventDTO)]

            trace = TurnTrace(
                intent=nlu_result.intent,
                story_matched=story_name,
                fallback_reason=fallback_reason,
                actions_executed=actions_executed,
                slot_changes={k: v for k, v in state.slots.items() if v != slots_before.get(k)},
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            logger.debug(f"Trace: {trace}")
            ctx.outgoing_events += tracker.get_thread_events_envelope()
            if bot_utters:
                for utter in bot_utters:
                    if utter.text:
                        ctx.send_message_with_buttons(message=utter.text, buttons=utter.buttons or [])

