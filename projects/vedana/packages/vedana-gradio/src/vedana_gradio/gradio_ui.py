import datetime
import io
import logging
import re
import traceback
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import gradio as gr
import pandas as pd
from jims_core.thread.thread_controller import ThreadController
from opentelemetry import trace
from uuid_extensions import uuid7
from vedana_core.data_model import DataModel
from vedana_core.data_provider import GristSQLDataProvider
from vedana_core.graph import Graph
from vedana_core.importers.fast import DataModelLoader
from vedana_core.rag_pipeline import RagPipeline

# todo
from vedana_core.settings import settings as s

tracer = trace.get_tracer(__name__)


class MemLogger(logging.Logger):
    """Logger that captures logs to a string buffer for debugging purposes."""

    def __init__(self, name: str, level: int = 0) -> None:
        super().__init__(name, level)
        self.parent = logging.getLogger(__name__)
        self._buf = io.StringIO()
        handler = logging.StreamHandler(self._buf)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        self.addHandler(handler)

    def get_logs(self) -> str:
        return self._buf.getvalue()

    def clear(self) -> None:
        self._buf.truncate(0)
        self._buf.seek(0)


logger = logging.getLogger(__name__)

# Global async event loop that runs in a separate thread
executor = ThreadPoolExecutor(max_workers=1)


class GlobalState:
    data_model: DataModel
    graph: Graph
    pipeline: RagPipeline


_global_state = GlobalState()


def reload_data_model(show_debug: bool = True) -> tuple[str, str]:
    """Reload data model and return updated UI components"""
    logger = MemLogger("reload_data_model", level=logging.DEBUG)
    data_model_text = ""

    try:
        logger.info("Starting data model reload")
        new_data_model = DataModel.load_grist_online(
            s.grist_data_model_doc_id, grist_server=s.grist_server_url, api_key=s.grist_api_key
        )
        logger.info("Data model loaded from Grist")

        _global_state.data_model = new_data_model
        result = (
            f"Successfully reloaded DataModel: "
            f"\n{len(new_data_model.anchors)} anchors;"
            f"\n{len(new_data_model.attrs)} attributes;"
            f"\n{len(new_data_model.links)} links."
        )
        logger.info(
            f"Data model updated: "
            f"{len(new_data_model.anchors)} anchors, "
            f"{len(new_data_model.attrs)} attributes, "
            f"{len(new_data_model.links)} links"
        )

        # Update data model text box with the new content
        if _global_state.data_model:
            data_model_text = _global_state.data_model.to_text_descr()

        # Add debug logs if requested
        if show_debug:
            result = f"{result}\n\nDebug Logs:\n{logger.get_logs()}"

    except Exception as e:
        logger.exception(f"Error reloading DataModel: {str(e)}")
        result = f"Error reloading DataModel: {str(e)}"
        if show_debug:
            result = f"{result}\n\nDebug Logs:\n{logger.get_logs()}"

    try:
        DataModelLoader(_global_state.data_model, _global_state.graph).update_data_model_node()
    except Exception as exc:
        logger.warning(f"Failed to store DataModel in graph: {exc}")

    return result, data_model_text


def parse_query_costs(model_usage: dict) -> list[dict]:
    # cost per 1M tokens - https://platform.openai.com/docs/pricing
    model_prices = {
        "gpt-4.1": {"prompt_tokens": 2, "cached_tokens": 0.5, "completion_tokens": 8},
        "gpt-4.1-mini": {"prompt_tokens": 0.4, "cached_tokens": 0.1, "completion_tokens": 1.6},
        "gpt-4.1-nano": {"prompt_tokens": 0.1, "cached_tokens": 0.025, "completion_tokens": 0.4},
        "gpt-4o": {"prompt_tokens": 2.5, "cached_tokens": 1.25, "completion_tokens": 10},
        "gpt-4o-mini": {"prompt_tokens": 0.15, "cached_tokens": 0.075, "completion_tokens": 0.6},
        "o4-mini": {"prompt_tokens": 1.1, "cached_tokens": 0.275, "completion_tokens": 4.4},
    }
    # cost per 1M --> cost per token
    model_prices = {mk: {k: v / 1e6 for k, v in mv.items()} for mk, mv in model_prices.items()}

    # actual model_id = re.sub(r'-\d{4}-\d{2}-\d{2}$', '', s)  # "gpt-4.1-2025-04-14" --> "gpt-4.1"
    model_usage = {re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model): v for model, v in model_usage.items()}

    model_usage_list = [
        {"model": m}
        | v
        | {
            "prompt_tokens, $": round(
                (v["prompt_tokens"] - v["cached_tokens"]) * model_prices.get(m, {}).get("prompt_tokens", 0), 4
            ),
            "cached_tokens, $": round(v["cached_tokens"] * model_prices.get(m, {}).get("cached_tokens", 0), 4),
            "completion_tokens, $": round(
                v["completion_tokens"] * model_prices.get(m, {}).get("completion_tokens", 0), 4
            ),
            "total, $": round(
                (v["prompt_tokens"] - v["cached_tokens"]) * model_prices.get(m, {}).get("prompt_tokens", 0)
                + v["cached_tokens"] * model_prices.get(m, {}).get("cached_tokens", 0)
                + v["completion_tokens"] * model_prices.get(m, {}).get("completion_tokens", 0),
                4,
            ),
        }
        for m, v in model_usage.items()
    ]
    return model_usage_list


async def process_query(
    text_query: str,
    show_debug: bool,
    thread_controller: ThreadController,
    pipeline: RagPipeline,
) -> tuple[str, str, str, str, str, dict]:
    text_query = text_query.strip()
    if not text_query.strip():
        return "", "", "", "", "", {}
    vts_res = ""
    tct_tech_res = ""
    tct_human_res = ""
    all_human_res = ""
    model_usage = {}

    # Create a memory logger to capture debug output
    logger = MemLogger("rag_debug", level=logging.DEBUG)
    pipeline.logger = logger  # pass this logger to pipeline, to retrieve logs for query

    try:
        await thread_controller.store_user_message(uuid7(), text_query)  # type: ignore

        events = await thread_controller.run_pipeline_with_context(pipeline)

        for event in events:
            if event.event_type == "comm.assistant_message":
                all_human_res = event.event_data.get("content", "")
                logger.info(f"Assistant message:\n{all_human_res}")
            elif event.event_type == "rag.query_processed":
                tech_info = event.event_data.get("technical_info", {})
                tct_tech_res = (
                    f"VTS Queries:\n{'\n'.join(tech_info.get('vts_queries', []))}\n"
                    f"Cypher Queries:\n{';\n'.join(tech_info.get('cypher_queries', []))}"
                )
                tct_human_res = event.event_data.get("answer", "")
                model_usage = tech_info.get("model_stats", {})
                logger.info(f"VTS queries:\n{'\n'.join(tech_info.get('vts_queries', []))}")
                logger.info(f"Cypher queries:\n{';\n'.join(tech_info.get('cypher_queries', []))}")
                logger.info(f"Model usage: {model_usage}")

        return vts_res, tct_tech_res, tct_human_res, all_human_res, logger.get_logs() if show_debug else "", model_usage

    except Exception as e:
        logger.exception(f"Error processing query: {e}")
        error_msg = f"Error: {str(e)}"
        debug_logs = logger.get_logs() if show_debug else ""
        if show_debug:
            debug_logs += f"\n\nTraceback:\n{traceback.format_exc()}"
        return "", "", error_msg, error_msg, debug_logs, model_usage


async def create_gradio_interface(graph: Graph, data_model: DataModel, sessionmaker) -> gr.Blocks:
    """Gradio interface with JIMS integration"""

    # Store in global state for reload functions
    _global_state.data_model = data_model
    _global_state.graph = graph

    # Initialize pipeline
    _global_state.pipeline = RagPipeline(
        graph=graph,
        data_model=data_model,
        logger=MemLogger("rag_pipeline", level=logging.DEBUG),
        threshold=0.8,
    )

    # Function to create a new thread controller for a new session
    async def init_thread_controller() -> ThreadController:
        thread_controller = await ThreadController.new_thread(
            sessionmaker,
            uuid7(),  # type: ignore
            {
                "interface": "gradio",
                "created_at": str(datetime.datetime.now()),
                "session_id": str(uuid7()),
            },
        )

        logger.info(f"Created new thread with ID: {thread_controller.thread.thread_id}")
        return thread_controller

    with gr.Blocks(title="Vedana Demo") as iface:
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown("# Vedana Demo")

        # Create a state component to store the thread controller for this session
        thread_controller_state = gr.State(value=None)

        with gr.Row():
            nl_input = gr.Textbox(
                lines=2,
                placeholder="Enter your question here...",
                label="Query",
            )

        submit_btn_text = gr.Button("Submit")

        with gr.Accordion("Settings", open=True):
            with gr.Row():
                reload_model_btn = gr.Button("Reload Data Model")
                clear_history_btn = gr.Button("Clear Conversation History")

            with gr.Row():
                with gr.Column():
                    show_debug = gr.Checkbox(label="Show Debug Output", value=True)

                    if s.debug:
                        available_models = list(
                            {
                                "gpt-5",
                                "gpt-5-mini",
                                "gpt-5-nano",
                                "gpt-4.1",
                                "gpt-4.1-mini",
                                "gpt-4.1-nano",
                                "gpt-4o",
                                "gpt-4o-mini",
                                "o4-mini",
                                s.model,
                            }
                        )

                        model_selector: gr.Component = gr.Dropdown(
                            choices=available_models,
                            label="Select LLM",
                            value=s.model,  # default = env
                            multiselect=False,
                        )

                    else:
                        model_selector = gr.State(value=s.model)

            with gr.Accordion("Data model", open=False):
                data_model_textbox = gr.Markdown(
                    value=data_model.to_text_descr(),
                    label="Text description",
                    show_copy_button=True,
                    max_height=400,
                )

        with gr.Row():
            vts_output = gr.Textbox(
                lines=4,
                label="Vector text search",
                autoscroll=False,
                show_copy_button=True,
            )
        with gr.Row():
            technical_output = gr.Textbox(
                lines=4,
                label="Technical Answer (Query and Raw Result)",
                autoscroll=False,
                show_copy_button=True,
            )
            human_output = gr.Textbox(
                lines=4,
                label="Human-Readable Query Answer",
                autoscroll=False,
                show_copy_button=True,
            )

        with gr.Row():
            human_output_tools = gr.Textbox(
                lines=4,
                label="Final Answer",
                autoscroll=False,
                show_copy_button=True,
            )

        # Add conversation history output
        history_output = gr.Textbox(
            lines=10,
            label="Conversation History",
            visible=True,
            show_copy_button=True,
        )

        debug_output = gr.Textbox(
            lines=10,
            label="Debug Output",
            visible=True,
            show_copy_button=True,
        )

        with gr.Row():  # Session stats
            session_info = gr.Textbox(label="Session ID", visible=True, interactive=False, scale=2)
            last_query_token_stats = gr.State(value={})
            # if s.debug:
            token_usage = gr.Dataframe(
                headers=[
                    "query",
                    "requests_count",
                    "prompt_tokens",
                    "prompt_tokens, $",
                    "cached_tokens",
                    "cached_tokens, $",
                    "completion_tokens",
                    "completion_tokens, $",
                    "total, $",
                ],
                datatype=["str", "number", "number", "number", "number", "number", "number", "number", "number"],
                col_count=(9, "fixed"),
                interactive=False,
                visible=True,
                scale=7,
            )
            # else:
            #     token_usage = gr.State(value=pd.DataFrame())

        # Ensure we have a thread controller for this session
        async def ensure_thread_controller(thread_controller) -> tuple[ThreadController, str]:
            if thread_controller is None:
                thread_controller = await init_thread_controller()
            return (
                thread_controller,
                f"Session ID: {thread_controller.thread.thread_config.get('session_id', 'unknown')}",
            )

        # Synchronous wrapper for async process_query
        async def process_query_sync(
            text_query,
            show_debug,
            selected_model,
            thread_controller,
        ):
            with tracer.start_as_current_span("gradio_ui.process_query_sync"):
                # Initialize thread_controller if needed
                if thread_controller is None:
                    thread_controller = await init_thread_controller()

                pipeline = _global_state.pipeline

                # Update pipeline with current data model and settings
                pipeline.data_model = _global_state.data_model

                if s.debug:  # pass selected model if app set to debug=True
                    pipeline.model = selected_model

                logger.info(f"Processing query: {text_query}")

                try:
                    result = await process_query(
                        text_query,
                        show_debug,
                        thread_controller,
                        pipeline,
                    )
                    return (thread_controller,) + result
                except Exception as e:
                    logger.error(f"Error in process_query_sync: {str(e)}")
                    return (
                        thread_controller,
                        "",
                        "",
                        f"Error: {str(e)}",
                        f"Error: {str(e)}",
                        traceback.format_exc() if show_debug else "",
                        {},
                    )

        # Function to get conversation history
        async def get_conversation_history(ctl: ThreadController) -> str:
            if ctl is None:
                return "No conversation history yet."

            ctx = await ctl.make_context()
            history_text = ""
            for event in ctx.history:
                role = "User" if event.get("role") == "user" else "Assistant"
                history_text += f"{role}: {event.get('content')}\n\n----------\n\n"
            return history_text

        # Function to clear conversation history
        async def clear_conversation_history(ctl: ThreadController) -> tuple[ThreadController, str]:
            # Create a new thread with a new ID
            new_thr_id: UUID = uuid7()  # type: ignore
            session_id = str(uuid7())
            new_controller = await ThreadController.new_thread(
                ctl.sessionmaker,
                new_thr_id,
                {
                    "interface": "gradio",
                    "created_at": str(datetime.datetime.now()),
                    "session_id": session_id,
                },
            )
            # Return new controller and empty history
            return (
                new_controller,
                f"Conversation history cleared. New thread (ID: {new_thr_id}) with session {session_id} created.",
            )

        # Synchronous wrapper for async get_conversation_history
        async def get_history_sync(thread_controller) -> str:
            if thread_controller is None:
                return "No conversation history yet."

            return await get_conversation_history(thread_controller)

        def get_llm_use_sync(session_tokens: pd.DataFrame, request_tokens: dict, request_query: str) -> pd.DataFrame:
            """Update total token usage per gradio instance / session"""
            new_row_list = parse_query_costs(request_tokens)
            new_row = pd.DataFrame(new_row_list)
            new_row["query"] = request_query

            if session_tokens.shape[0] > 1:
                session_tokens = session_tokens.head(session_tokens.shape[0] - 1)  # remove previous "Total:"

            session_df = pd.concat([session_tokens, new_row], ignore_index=True)
            if session_df.shape[0] > 0:
                sum_row = session_df.sum(axis=0)
                sum_row["model"] = "-"
                sum_row["query"] = "Total:"
                session_df = pd.concat([session_df, pd.DataFrame(sum_row).T], ignore_index=True)
            return session_df

        async def clear_history_sync(thread_controller) -> tuple:
            if thread_controller is None:
                thread_controller = await init_thread_controller()

            new_controller, message = await clear_conversation_history(thread_controller)
            return (
                new_controller,
                message,
                f"Session ID: {new_controller.thread.thread_config.get('session_id', 'unknown')}",
                "",
                "",
                "",
                "",
                "",
                pd.DataFrame(),
            )

        # Submit button click
        submit_btn_text.click(
            fn=process_query_sync,
            inputs=[
                nl_input,
                show_debug,
                model_selector,
                thread_controller_state,
            ],
            outputs=[
                thread_controller_state,
                vts_output,
                technical_output,
                human_output,
                human_output_tools,
                debug_output,
                last_query_token_stats,
            ],
        ).then(
            fn=get_history_sync,
            inputs=[thread_controller_state],
            outputs=[history_output],
        ).then(
            fn=get_llm_use_sync,
            inputs=[token_usage, last_query_token_stats, nl_input],
            outputs=[token_usage],
        )

        # Reload data model button click
        reload_model_btn.click(
            fn=reload_data_model,
            inputs=[show_debug],
            outputs=[debug_output, data_model_textbox],
        )

        # Clear history button click
        clear_history_btn.click(
            fn=clear_history_sync,
            inputs=[thread_controller_state],
            outputs=[
                thread_controller_state,
                history_output,
                session_info,
                debug_output,
                vts_output,
                technical_output,
                human_output,
                human_output_tools,
                token_usage,
            ],
        )

        # Initialize on page load: create a new thread controller for this session
        iface.load(
            fn=ensure_thread_controller,
            inputs=[thread_controller_state],
            outputs=[thread_controller_state, session_info],
        ).then(
            fn=get_history_sync,
            inputs=[thread_controller_state],
            outputs=[history_output],
        )

    return iface
