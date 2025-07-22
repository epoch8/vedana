import asyncio
import datetime
import io
import logging
import re
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import gradio as gr
import pandas as pd
from jims_core.thread.thread_controller import ThreadController
from opentelemetry import trace
from uuid_extensions import uuid7

from vedana.data_model import DataModel
from vedana.data_provider import GristSQLDataProvider
from vedana.embeddings import EmbeddingProvider, OpenaiEmbeddingProvider
from vedana.graph import Graph, MemgraphGraph
from vedana.importers.fast import DataModelLoader, update_graph
from vedana.rag_pipeline import RagPipeline
from vedana.settings import get_custom_settings
from vedana.settings import settings as s

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
loop = None


class GlobalState:
    data_model: DataModel
    embed_provider: EmbeddingProvider
    graph: Graph
    pipeline: RagPipeline


_global_state = GlobalState()


def start_background_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def init_async_stuff():
    global loop
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=start_background_loop, args=(loop,), daemon=True)
    t.start()


def reload_graph(show_debug: bool = True) -> str:
    """Reload graph data from Grist."""
    logger = MemLogger("reload_graph", level=logging.DEBUG)

    if not s.grist_data_doc_id:
        return "Error: GRIST_DATA_DOC_ID environment variable is not set. Cannot reload graph data."
    if not _global_state.data_model:
        return "Error: Data model not loaded. Reload data model first."
    if not _global_state.embed_provider:
        return "Error: Embedding provider not available."

    try:
        graph = _global_state.graph
        logger.info("Starting graph reload process")

        try:
            n_nodes = graph.number_of_nodes()
            n_edges = graph.number_of_edges()
            logger.info(f"Current nodes count: {n_nodes}; current edges count: {n_edges}")
        except Exception as e:
            logger.warning(f"Error parsing current graph configuration: {str(e)}")

        data_provider = GristSQLDataProvider(
            doc_id=s.grist_data_doc_id, grist_server=s.grist_server_url, api_key=s.grist_api_key
        )
        logger.info("Created data provider")

        with data_provider:
            logger.info("Starting multiprocess graph update")
            update_graph(
                graph=graph,
                dp=data_provider,
                data_model=_global_state.data_model,
                embed_provider=_global_state.embed_provider,
                dry_run=False,
                node_batch_size=200,
                edge_batch_size=100,
            )

        success_msg = "Successfully reloaded graph data from Grist"
        logger.info(success_msg)

        if show_debug:
            return f"{success_msg}\n\nDebug Logs:\n{logger.get_logs()}"
        return success_msg

    except Exception as e:
        error_msg = f"Error reloading graph data: {str(e)}"
        logger.exception(error_msg)
        if show_debug:
            return f"{error_msg}\n\nDebug Logs:\n{logger.get_logs()}"
        return error_msg


def reload_data_model(current_selected_vts_props: list, show_debug: bool = True) -> tuple[str, str, dict[str, Any]]:
    """Reload data model and return updated UI components"""
    logger = MemLogger("reload_data_model", level=logging.DEBUG)
    data_model_text = ""
    new_vts_props = []

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

            # Update the VTS properties dropdown
            new_vts_props = sorted(
                [f"{v['noun']}::{name}" for name, v in _global_state.data_model.embeddable_attributes().items()]
            )
            logger.info(f"Generated {len(new_vts_props)} VTS properties")

        # Add debug logs if requested
        if show_debug:
            result = f"{result}\n\nDebug Logs:\n{logger.get_logs()}"

    except Exception as e:
        logger.exception(f"Error reloading DataModel: {str(e)}")
        result = f"Error reloading DataModel: {str(e)}"
        if show_debug:
            result = f"{result}\n\nDebug Logs:\n{logger.get_logs()}"

    current_selected_vts_props = [e for e in current_selected_vts_props if e in new_vts_props]

    try:
        DataModelLoader(_global_state.data_model, _global_state.graph).update_data_model_node()
    except Exception as exc:
        logger.warning(f"Failed to store DataModel in graph: {exc}")

    return result, data_model_text, gr.update(choices=new_vts_props, value=current_selected_vts_props)


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

    model_usage = [
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
    return model_usage


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
        await thread_controller.store_user_message(uuid7(), text_query)

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


def load_data_source(selected_project: str = None):
    if not s.debug:
        return "Error updating project config: debug set to False", "", ""

    # update envs
    ProjectConfig = get_custom_settings(selected_project.upper() + "_")
    project_settings = ProjectConfig()

    # data model
    s.grist_data_model_doc_id = project_settings.grist_data_model_doc_id
    s.grist_server_url = project_settings.grist_server_url
    s.grist_api_key = project_settings.grist_api_key

    debug_output, data_model_textbox, vts_props = reload_data_model(current_selected_vts_props=[], show_debug=True)

    # graph data path (for reloads)
    s.grist_data_doc_id = project_settings.grist_data_doc_id

    # memgraph connection
    s.memgraph_uri = project_settings.memgraph_uri
    s.memgraph_user = project_settings.memgraph_user
    s.memgraph_pwd = project_settings.memgraph_pwd
    _global_state.graph = MemgraphGraph(s.memgraph_uri, s.memgraph_user, s.memgraph_pwd)

    # embeds (for reloads)
    s.embeddings_cache_path = project_settings.embeddings_cache_path or s.default_embeddings_cache_path
    s.embeddings_dim = project_settings.embeddings_dim
    _global_state.embed_provider = OpenaiEmbeddingProvider(s.embeddings_cache_path, s.embeddings_dim)

    # Re-initialize pipeline
    _global_state.pipeline = RagPipeline(
        graph=_global_state.graph,
        embed_provider=_global_state.embed_provider,
        data_model=_global_state.data_model,
        logger=MemLogger("rag_pipeline", level=logging.DEBUG),
        threshold=0.8,
        temperature=0.0,
    )

    # cache data model
    try:
        DataModelLoader(_global_state.data_model, _global_state.graph).update_data_model_node()
    except Exception as cache_exc:
        logger.warning(f"Failed to cache DataModel: {cache_exc}")

    return debug_output, data_model_textbox, vts_props


def create_gradio_interface(graph: Graph, embed_provider: EmbeddingProvider, data_model: DataModel, sessionmaker, loop):
    """Gradio interface with JIMS integration"""

    # Store in global state for reload functions
    _global_state.data_model = data_model
    _global_state.embed_provider = embed_provider
    _global_state.graph = graph

    # Initialize pipeline
    _global_state.pipeline = RagPipeline(
        graph=graph,
        embed_provider=embed_provider,
        data_model=data_model,
        logger=MemLogger("rag_pipeline", level=logging.DEBUG),
        threshold=0.8,
        temperature=0.0,
    )

    # Function to create a new thread controller for a new session
    def init_thread_controller():
        # Create a future to run the async function
        future = asyncio.run_coroutine_threadsafe(
            ThreadController.new_thread(
                sessionmaker,
                uuid7(),
                {
                    "interface": "gradio",
                    "created_at": str(datetime.datetime.now()),
                    "session_id": str(uuid7()),
                },
            ),
            loop,
        )
        # Wait for the result
        thread_controller = future.result(timeout=10)
        logger.info(f"Created new thread with ID: {thread_controller.thread.thread_id}")
        return thread_controller

    with gr.Blocks(title="Vedana Demo") as iface:
        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                gr.Markdown("# Vedana Demo")

            if s.debug:
                # todo move available projects to env
                available_projects = ["Corestone", "Maytoni", "MaytoniV2", "NL", "Solar", "Agenyz", "Formelskin"]
                project_id = gr.Dropdown(
                    choices=available_projects,
                    label="Select Project...",
                    value="Maytoni",  # default = from env
                    multiselect=False,
                    interactive=True,
                    scale=1,
                )
            else:
                project_id = gr.State(value=None)

            sync_project_id = gr.Button("Update Project", visible=s.debug, interactive=s.debug, scale=1)

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
                reload_graph_btn = gr.Button("Reload Graph Data", interactive=s.debug, visible=s.debug)
                clear_history_btn = gr.Button("Clear Conversation History")

            # Confirmation UI for graph reload
            with gr.Row(visible=False) as confirm_reload_graph_row:
                gr.Markdown(
                    "⚠️ **Confirm Graph Reload** ⚠️\n\nThis is a slow and resource-expensive operation. Are you sure you want to proceed?"
                )
                with gr.Column(scale=1):  # Using a column for button arrangement
                    confirm_reload_graph_yes_btn = gr.Button("✅ Yes, Reload Now")
                    confirm_reload_graph_cancel_btn = gr.Button("❌ Cancel")

            with gr.Row():
                with gr.Column():
                    show_debug = gr.Checkbox(label="Show Debug Output", value=True)

                    if s.debug:
                        # https://platform.openai.com/docs/models
                        # Можно распарсить все модели из API:
                        # from openai import OpenAI
                        # client = OpenAI()
                        # models = client.models.list()
                        # available_models = [
                        #     model.id for model in models.data if "gpt" in model.id and not model.id.startswith("ft:")
                        # ]
                        available_models = [
                            "gpt-4.1",
                            "gpt-4.1-mini",
                            "gpt-4.1-nano",
                            "gpt-4o",
                            "gpt-4o-mini",
                            "o4-mini",
                        ]

                        model_selector = gr.Dropdown(
                            choices=available_models,
                            label="Select LLM",
                            value=s.model,  # default = env
                            multiselect=False,
                        )

                    else:
                        model_selector = gr.State(value=s.model)

                with gr.Column():
                    tct_temperature = gr.Number(0.01, label="Temperature", minimum=0, maximum=1, step=0.05)
                with gr.Column():
                    use_vector_text_search = gr.Checkbox(label="Use vector text search", value=False)
                    vts_props = sorted([f"{v['noun']}::{n}" for n, v in data_model.embeddable_attributes().items()])
                    vector_text_search_props = gr.Dropdown(
                        vts_props,
                        label="Search props",
                        multiselect=True,
                        value=vts_props[0] if vts_props else None,
                    )
                    vts_threshold = gr.Number(0.8, label="Threshold", minimum=0, maximum=1, step=0.05)
                    vts_top_n = gr.Number(5, label="Top-N Results", minimum=1, maximum=30, step=1, precision=0)
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
        def ensure_thread_controller(thread_controller):
            if thread_controller is None:
                thread_controller = init_thread_controller()
            return (
                thread_controller,
                f"Session ID: {thread_controller.thread.thread_config.get('session_id', 'unknown')}",
            )

        # Synchronous wrapper for async process_query
        def process_query_sync(
            text_query,
            show_debug,
            use_vector_text_search,
            vts_threshold,
            vts_top_n,
            tct_temperature,
            selected_model,
            thread_controller,
        ):
            with tracer.start_as_current_span("gradio_ui.process_query_sync"):
                # Initialize thread_controller if needed
                if thread_controller is None:
                    thread_controller = init_thread_controller()

                pipeline = _global_state.pipeline

                # Update pipeline with current data model and settings
                pipeline.data_model = _global_state.data_model
                pipeline.threshold = vts_threshold
                pipeline.temperature = tct_temperature
                if use_vector_text_search:
                    pipeline.top_n = vts_top_n
                if s.debug:  # pass selected model if app set to debug=True
                    pipeline.model = selected_model

                logger.info(f"Processing query: {text_query}")
                logger.info(
                    f"Pipeline run settings:\n "
                    f"- VTS: {use_vector_text_search}, threshold: {vts_threshold}, n={pipeline.top_n};\n"
                    f" temperature: {tct_temperature};"
                )

                # Use the global event loop
                future = asyncio.run_coroutine_threadsafe(
                    process_query(
                        text_query,
                        show_debug,
                        thread_controller,
                        pipeline,
                    ),
                    loop,
                )

                try:
                    result = future.result(timeout=120)
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
                role = "User" if event["role"] == "user" else "Assistant"
                history_text += f"{role}: {event['content']}\n\n----------\n\n"
            return history_text

        # Function to clear conversation history
        async def clear_conversation_history(ctl: ThreadController) -> tuple[ThreadController, str]:
            # Create a new thread with a new ID
            new_thr_id = uuid7()
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
        def get_history_sync(thread_controller):
            if thread_controller is None:
                return "No conversation history yet."

            future = asyncio.run_coroutine_threadsafe(get_conversation_history(thread_controller), loop)
            return future.result(timeout=10)

        def get_llm_use_sync(session_tokens: pd.DataFrame, request_tokens: dict, request_query: str) -> pd.DataFrame:
            """Update total token usage per gradio instance / session"""
            new_row = parse_query_costs(request_tokens)
            new_row = pd.DataFrame(new_row)
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

        def clear_history_sync(thread_controller):
            if thread_controller is None:
                thread_controller = init_thread_controller()

            future = asyncio.run_coroutine_threadsafe(clear_conversation_history(thread_controller), loop)
            new_controller, message = future.result(timeout=10)
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

        # Helper functions for graph reload confirmation
        def handle_show_reload_confirmation():
            return gr.update(visible=True)

        def handle_cancel_reload_confirmation():
            return gr.update(visible=False)

        def handle_confirm_reload_graph(show_debug):
            # This function will call reload_graph and then hide the confirmation
            debug_msg = reload_graph(show_debug)  # reload_graph is globally defined
            return debug_msg, gr.update(visible=False)

        # Submit button click
        submit_btn_text.click(
            fn=process_query_sync,
            inputs=[
                nl_input,
                show_debug,
                use_vector_text_search,
                vts_threshold,
                vts_top_n,
                tct_temperature,
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

        # Change data source (debug app)
        sync_project_id.click(
            fn=load_data_source,
            inputs=[project_id],
            outputs=[debug_output, data_model_textbox, vector_text_search_props],
        ).then(
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

        # Reload data model button click
        reload_model_btn.click(
            fn=reload_data_model,
            inputs=[vector_text_search_props, show_debug],
            outputs=[debug_output, data_model_textbox, vector_text_search_props],
        )

        if s.debug:
            reload_graph_btn.click(fn=handle_show_reload_confirmation, inputs=[], outputs=[confirm_reload_graph_row])

            # Handlers for the graph reload confirmation buttons
            confirm_reload_graph_yes_btn.click(
                fn=handle_confirm_reload_graph,
                inputs=[show_debug],
                outputs=[
                    debug_output,
                    confirm_reload_graph_row,
                ],  # debug_output first for the message, then the row to hide
            )

        confirm_reload_graph_cancel_btn.click(
            fn=handle_cancel_reload_confirmation, inputs=[], outputs=[confirm_reload_graph_row]
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
