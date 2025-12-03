import reflex as rx

from vedana_backoffice.states.eval import EvalState
from vedana_backoffice.ui import app_header


def _selection_and_actions() -> rx.Component:
    """Selection controls and action buttons for the right panel."""
    return rx.vstack(
        rx.hstack(
            rx.text(EvalState.selection_label, weight="medium"),
            rx.badge(EvalState.cost_label, color_scheme="gray", variant="soft"),
            rx.spacer(),
            rx.button(
                "Reset selection",
                variant="ghost",
                color_scheme="gray",
                size="1",
                disabled=rx.cond(EvalState.selected_count > 0, False, True),  # type: ignore[arg-type]
                on_click=EvalState.reset_selection,
            ),
            align="center",
            width="100%",
        ),
        rx.hstack(
            rx.button(
                "Run selected",
                color_scheme="blue",
                on_click=EvalState.run_selected_tests,
                loading=EvalState.is_running,
                disabled=rx.cond(EvalState.can_run, False, True),  # type: ignore[arg-type]
            ),
            rx.button(
                "Refresh Data Model",
                variant="soft",
                on_click=EvalState.refresh_data_model,
                loading=EvalState.loading,
            ),
            rx.button(
                "Refresh Judge Config",
                variant="soft",
                on_click=EvalState.run_judge_refresh,
                loading=EvalState.is_running,
            ),
            spacing="3",
        ),
        rx.button(
            "Reload data",
            variant="soft",
            color_scheme="gray",
            size="1",
            on_click=EvalState.load_eval_data,
            loading=EvalState.loading,
            width="100%",
        ),
        spacing="3",
        width="100%",
    )


def _questions_card() -> rx.Component:
    def _row(row: dict[str, rx.Var]) -> rx.Component:
        return rx.table.row(
            rx.table.cell(
                rx.checkbox(
                    checked=row.get("selected", False),
                    on_change=EvalState.toggle_question_selection(question=row.get("id", "")),  # type: ignore[arg-type]
                )
            ),
            rx.table.cell(rx.text(row.get("gds_question", ""), weight="medium")),
            rx.table.cell(
                rx.text(
                    row.get("gds_answer", ""),
                    size="2",
                )
            ),
            rx.table.cell(
                rx.text(
                    row.get("question_context", ""),
                    size="1",
                    color="gray",
                )
            ),
            rx.table.cell(
                rx.cond(
                    row.get("question_scenario", "") != "",
                    rx.badge(
                        row.get("question_scenario", ""),
                        variant="soft",
                        size="1",
                        color_scheme=row.get("scenario_color", "gray"),
                    ),
                    rx.box(),
                )
            ),
        )

    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading("Golden QA Dataset", size="4"),
                rx.spacer(),
                rx.hstack(
                    rx.text("Scenario", size="1", color="gray"),
                    rx.select(
                        items=EvalState.available_scenarios,
                        value=EvalState.selected_scenario,
                        on_change=EvalState.set_scenario,
                        width="12em",
                    ),
                    spacing="2",
                    align="center",
                ),
                align="center",
                width="100%",
            ),
            rx.scroll_area(
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell(
                                rx.checkbox(
                                    checked=EvalState.all_selected,
                                    on_change=EvalState.toggle_select_all,
                                )
                            ),
                            rx.table.column_header_cell("Question"),
                            rx.table.column_header_cell("Golden answer"),
                            rx.table.column_header_cell("Context"),
                            rx.table.column_header_cell("Scenario"),
                        )
                    ),
                    rx.table.body(rx.foreach(EvalState.eval_gds_rows_with_selection, _row)),
                    variant="surface",
                    style={"width": "100%", "tableLayout": "fixed"},
                ),
                type="always",
                scrollbars="vertical",
                style={"flex": "1", "width": "100%", "minHeight": "0"},
            ),
            spacing="3",
            style={"height": "100%", "display": "flex", "flexDirection": "column"},
        ),
        padding="1em",
        width="100%",
        style={"height": "100%", "display": "flex", "flexDirection": "column"},
    )


def _judge_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Judge configuration", size="4"),
            rx.select(
                items=EvalState.judge_model_options,
                value=EvalState.selected_judge_model,
                placeholder="No judge models",
                on_change=EvalState.set_judge_model,
            ),
            rx.button(
                "View Judge Prompt",
                variant="soft",
                size="1",
                on_click=EvalState.open_judge_prompt_dialog,
                disabled=rx.cond(EvalState.judge_prompt_id != "", False, True),  # type: ignore[arg-type]
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        padding="1em",
        width="100%",
    )


def _pipeline_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Pipeline config", size="4"),
            rx.vstack(
                rx.box(
                    rx.text("Data model", weight="medium"),
                    rx.text(f"Snapshot @ {EvalState.dm_snapshot_updated}", size="1", color="gray"),
                    rx.button(
                        "View Data Model",
                        variant="soft",
                        size="1",
                        on_click=EvalState.open_data_model_dialog,
                        disabled=rx.cond(EvalState.dm_id != "", False, True),  # type: ignore[arg-type]
                        width="100%",
                        margin_top="0.5em",
                    ),
                    padding_bottom="0.75em",
                ),
                rx.box(
                    rx.text("Pipeline model", weight="medium"),
                    rx.text(
                        rx.cond(EvalState.pipeline_model != "", EvalState.pipeline_model, "—"),
                        size="3",
                    ),
                    padding_bottom="0.75em",
                ),
                rx.box(
                    rx.text("Embeddings", weight="medium"),
                    rx.text(
                        rx.cond(EvalState.embeddings_model != "", EvalState.embeddings_model, "—"),
                        size="3",
                    ),
                    rx.text(
                        EvalState.embeddings_dim_label,
                        size="1",
                        color="gray",
                    ),
                ),
                spacing="1",
                width="100%",
            ),
            spacing="3",
            width="100%",
        ),
        padding="1em",
        width="100%",
    )


def _tests_card() -> rx.Component:
    def _row(row: dict[str, rx.Var]) -> rx.Component:
        return rx.table.row(
            rx.table.cell(rx.text(row.get("test_date", ""))),
            rx.table.cell(
                rx.text(
                    row.get("gds_question", ""),
                    weight="medium",
                )
            ),
            rx.table.cell(rx.text(row.get("pipeline_model", ""))),
            rx.table.cell(
                rx.text(
                    row.get("llm_answer", ""),
                    size="1",
                )
            ),
            rx.table.cell(
                rx.text(
                    row.get("gds_answer", ""),
                    size="1",
                )
            ),
            rx.table.cell(
                rx.badge(
                    row.get("test_status", ""),
                    color_scheme=row.get("status_color", "gray"),
                    variant="soft",
                )
            ),
            rx.table.cell(
                rx.text(
                    row.get("eval_judge_comment", ""),
                    size="1",
                    color="gray",
                )
            ),
        )

    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading("Latest test results", size="4"),
                rx.spacer(),
                align="center",
                width="100%",
            ),
            rx.scroll_area(
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("Run at"),
                            rx.table.column_header_cell("Question"),
                            rx.table.column_header_cell("Pipeline"),
                            rx.table.column_header_cell("Answer"),
                            rx.table.column_header_cell("Golden Answer"),
                            rx.table.column_header_cell("Status"),
                            rx.table.column_header_cell("Judge comment"),
                        )
                    ),
                    rx.table.body(rx.foreach(EvalState.tests_rows, _row)),
                    variant="surface",
                    style={"width": "100%", "tableLayout": "fixed"},
                ),
                type="always",
                scrollbars="vertical",
                style={"height": "360px", "width": "100%"},
            ),
            # Server-side pagination controls
            rx.hstack(
                rx.text(EvalState.tests_rows_display, size="2", color="gray"),  # type: ignore[arg-type]
                rx.spacer(),
                rx.hstack(
                    rx.button(
                        "⏮",
                        variant="soft",
                        size="1",
                        on_click=EvalState.tests_first_page,
                        disabled=~EvalState.tests_has_prev,  # type: ignore[operator]
                    ),
                    rx.button(
                        "← Prev",
                        variant="soft",
                        size="1",
                        on_click=EvalState.tests_prev_page,
                        disabled=~EvalState.tests_has_prev,  # type: ignore[operator]
                    ),
                    rx.text(
                        EvalState.tests_page_display,  # type: ignore[arg-type]
                        size="2",
                        style={"minWidth": "100px", "textAlign": "center"},
                    ),
                    rx.button(
                        "Next →",
                        variant="soft",
                        size="1",
                        on_click=EvalState.tests_next_page,
                        disabled=~EvalState.tests_has_next,  # type: ignore[operator]
                    ),
                    rx.button(
                        "⏭",
                        variant="soft",
                        size="1",
                        on_click=EvalState.tests_last_page,
                        disabled=~EvalState.tests_has_next,  # type: ignore[operator]
                    ),
                    spacing="2",
                    align="center",
                ),
                align="center",
                width="100%",
            ),
            spacing="3",
        ),
        padding="1em",
        width="100%",
    )


def _judge_prompt_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Judge Prompt"),
            rx.vstack(
                rx.text(f"Prompt ID: {EvalState.judge_prompt_id}", size="2", color="gray"),
                rx.box(
                    rx.text(
                        rx.cond(
                            EvalState.selected_judge_prompt != "",
                            EvalState.selected_judge_prompt,
                            "Prompt not loaded",
                        ),
                        size="2",
                    ),
                    padding="1em",
                    border="1px solid var(--gray-6)",
                    border_radius="8px",
                    style={"maxHeight": "60vh", "overflow": "auto", "whiteSpace": "pre-wrap"},
                ),
                rx.dialog.close(
                    rx.button("Close", variant="soft"),
                ),
                spacing="3",
                width="100%",
            ),
            style={"maxWidth": "800px"},
        ),
        open=EvalState.judge_prompt_dialog_open,
        on_open_change=EvalState.set_judge_prompt_dialog_open,
    )


def _data_model_dialog() -> rx.Component:
    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Data Model"),
            rx.vstack(
                rx.text(f"Model ID: {EvalState.dm_id}", size="2", color="gray"),
                rx.text(f"Snapshot @ {EvalState.dm_snapshot_updated}", size="2", color="gray"),
                rx.box(
                    rx.text(
                        rx.cond(
                            EvalState.dm_description != "",
                            EvalState.dm_description,
                            "Description not loaded",
                        ),
                        size="2",
                    ),
                    padding="1em",
                    border="1px solid var(--gray-6)",
                    border_radius="8px",
                    style={"maxHeight": "60vh", "overflow": "auto", "whiteSpace": "pre-wrap"},
                ),
                rx.dialog.close(
                    rx.button("Close", variant="soft"),
                ),
                spacing="3",
                width="100%",
            ),
            style={"maxWidth": "800px"},
        ),
        open=EvalState.data_model_dialog_open,
        on_open_change=EvalState.set_data_model_dialog_open,
    )


def _status_messages() -> rx.Component:
    return rx.vstack(
        rx.cond(
            EvalState.status_message != "",
            rx.callout(EvalState.status_message, color_scheme="green", variant="soft"),
            rx.box(),
        ),
        rx.cond(
            EvalState.error_message != "",
            rx.callout(EvalState.error_message, color_scheme="red", variant="soft"),
            rx.box(),
        ),
        rx.cond(
            EvalState.has_run_progress,
            rx.card(
                rx.vstack(
                    rx.heading("Run log", size="3"),
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(
                                EvalState.run_progress,
                                lambda line: rx.text(line, size="1"),
                            )
                        ),
                        type="always",
                        scrollbars="vertical",
                        style={"height": "160px"},
                    ),
                    spacing="2",
                ),
                padding="1em",
                width="100%",
            ),
            rx.box(),
        ),
        spacing="3",
        width="100%",
    )


def page() -> rx.Component:
    return rx.vstack(
        app_header(),
        rx.vstack(
            rx.grid(
                _questions_card(),
                rx.vstack(
                    _judge_card(),
                    _pipeline_card(),
                    _selection_and_actions(),
                    spacing="4",
                    width="100%",
                ),
                columns="2",
                spacing="4",
                width="100%",
                style={"gridTemplateColumns": "3fr 1fr", "height": "calc(100vh - 200px)", "minHeight": "600px"},
            ),
            _tests_card(),
            _status_messages(),
            spacing="4",
            width="100%",
        ),
        _judge_prompt_dialog(),
        _data_model_dialog(),
        align="start",
        spacing="2",
        padding="1.5em",
    )
