import reflex as rx

from vedana_backoffice.states.eval import EvalState, RunSummary
from vedana_backoffice.ui import app_header


def _selection_and_actions() -> rx.Component:
    """Selection controls and action buttons for the right panel."""
    return rx.vstack(
        rx.hstack(
            rx.input(
                placeholder="Test run name",
                value=EvalState.test_run_name,
                on_change=EvalState.set_test_run_name,
                width="18em",
            ),
            rx.button(
                EvalState.selection_label,
                color_scheme="blue",
                on_click=EvalState.run_selected_tests,
                loading=EvalState.is_running,
                disabled=rx.cond(EvalState.can_run, False, True),  # type: ignore[arg-type]
            ),
            rx.spacer(),
            rx.button(
                "Reset selection",
                variant="ghost",
                color_scheme="gray",
                size="1",
                disabled=rx.cond(EvalState.selected_count > 0, False, True),  # type: ignore[arg-type]
                on_click=EvalState.reset_selection,
            ),
            spacing="2",
            align="center",
            width="100%",
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
    def _expandable_text(row: dict[str, rx.Var], key: str, clamp: int = 10) -> rx.Component:
        """todo reuse _expandable_text in both tables"""
        row_id = row.get("id", "")
        return rx.table.cell(
            rx.box(
                rx.cond(
                    row.get("expanded", False),
                    rx.text(row.get(key, ""), size="1", white_space="pre-wrap"),
                    rx.text(
                        row.get(key, ""),
                        size="1",
                        white_space="pre-wrap",
                        style={
                            "display": "-webkit-box",
                            "WebkitLineClamp": str(clamp),
                            "WebkitBoxOrient": "vertical",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                        },
                    ),
                ),
                cursor="pointer",
                on_click=EvalState.toggle_gds_row(row_id=row_id),  # type: ignore[arg-type,call-arg,func-returns-value]
            )
        )

    def _row(row: dict[str, rx.Var]) -> rx.Component:
        return rx.table.row(
            rx.table.cell(
                rx.checkbox(
                    checked=row.get("selected", False),
                    on_change=EvalState.toggle_question_selection(question=row.get("id", "")),  # type: ignore[arg-type,call-arg,func-returns-value]
                )
            ),
            _expandable_text(row, "gds_question"),
            _expandable_text(row, "gds_answer"),
            _expandable_text(row, "question_context"),
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
                    rx.tooltip(
                        rx.button(
                            "↻",
                            variant="ghost",
                            color_scheme="gray",
                            size="1",
                            on_click=EvalState.refresh_golden_dataset,
                            loading=EvalState.is_running,
                        ),
                        content="Refresh golden dataset from Grist",
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
        style={"maxHeight": "80vh", "display": "flex", "flexDirection": "column"},
    )


def _judge_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Judge configuration", size="4"),
            rx.box(
                rx.text("Judge model", weight="medium"),
                rx.text(EvalState.judge_model, size="3"),
                padding_bottom="0.75em",
            ),
            rx.button(
                "View Judge Prompt",
                variant="soft",
                size="1",
                on_click=EvalState.open_judge_prompt_dialog,
                disabled=rx.cond(EvalState.judge_prompt_id != "", False, True),  # type: ignore[arg-type]
                width="100%",
                margin_top="0.5em",
            ),
            rx.button(
                "Refresh Judge Config",
                variant="soft",
                size="1",
                on_click=EvalState.run_judge_refresh,
                loading=EvalState.is_running,
                width="100%",
                margin_top="0.5em",
            ),
            spacing="1",
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
                    rx.button(
                        "View Data Model",
                        variant="soft",
                        size="1",
                        on_click=EvalState.open_data_model_dialog,
                        disabled=rx.cond(EvalState.dm_id != "", False, True),  # type: ignore[arg-type]
                        width="100%",
                        margin_top="0.5em",
                    ),
                    rx.button(
                        "Refresh Data Model",
                        variant="soft",
                        size="1",
                        on_click=EvalState.refresh_data_model,
                        loading=EvalState.loading,
                        width="100%",
                        margin_top="0.5em",
                    ),
                    width="100%",
                    padding_bottom="0.75em",
                ),
                rx.box(
                    rx.text("Pipeline model", weight="medium"),
                    rx.hstack(
                        rx.select(
                            items=["openai", "openrouter"],
                            value=EvalState.provider,
                            on_change=EvalState.set_provider,
                            width="100%",
                            placeholder="Provider",
                        ),
                        rx.cond(
                            EvalState.provider == "openrouter",
                            rx.input(
                                placeholder=rx.cond(
                                    EvalState.default_openrouter_key_present,
                                    "(Optional) custom OPENROUTER_API_KEY",
                                    "(Required) OPENROUTER_API_KEY",
                                ),
                                type="password",
                                value=EvalState.custom_openrouter_key,
                                on_change=EvalState.set_custom_openrouter_key,
                                width="100%",
                                required=rx.cond(EvalState.default_openrouter_key_present, False, True),
                            ),
                        ),
                        rx.select(
                            items=EvalState.available_models_view,
                            value=EvalState.pipeline_model,
                            on_change=EvalState.set_pipeline_model,
                            width="100%",
                            placeholder="Select model",
                        ),
                        spacing="2",
                        align="center",
                        wrap="wrap",
                        width="100%",
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
    def _expandable_text(row: dict[str, rx.Var], key: str, clamp: int = 2) -> rx.Component:
        row_id = row.get("row_id", "")
        return rx.table.cell(
            rx.box(
                rx.cond(
                    row.get("expanded", False),
                    rx.text(
                        row.get(key, ""),
                        size="1",
                        white_space="pre-wrap",
                        style={"wordBreak": "break-word"},
                    ),
                    rx.text(
                        row.get(key, ""),
                        size="1",
                        white_space="pre-wrap",
                        style={
                            "display": "-webkit-box",
                            "WebkitLineClamp": str(clamp),
                            "WebkitBoxOrient": "vertical",
                            "overflow": "hidden",
                            "textOverflow": "ellipsis",
                            "wordBreak": "break-word",
                        },
                    ),
                ),
                cursor="pointer",
                on_click=EvalState.toggle_row_expand(row_id=row_id),  # type: ignore[arg-type,call-arg,func-returns-value]
                style={"minWidth": "0", "width": "100%"},
            ),
            style={"minWidth": "0"},
        )

    def _row(row: dict[str, rx.Var]) -> rx.Component:
        return rx.table.row(
            rx.table.cell(rx.text(row.get("test_date", ""))),
            _expandable_text(row, "gds_question"),
            rx.table.cell(rx.text(row.get("pipeline_model", ""))),
            _expandable_text(row, "llm_answer"),
            _expandable_text(row, "gds_answer"),
            rx.table.cell(
                rx.badge(
                    row.get("test_status", ""),
                    color_scheme=row.get("status_color", "gray"),
                    variant="soft",
                )
            ),
            rx.table.cell(rx.text(row.get("eval_judge_rating", "—"))),
            _expandable_text(row, "eval_judge_comment"),
        )

    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.heading("Test results", size="4"),
                rx.spacer(),
                rx.text(EvalState.pass_fail_summary, size="2", color="gray"),
                rx.badge(EvalState.cost_label, color_scheme="gray", variant="soft"),
                rx.select(
                    items=EvalState.tests_sort_options,
                    value=EvalState.selected_tests_sort,
                    placeholder="Sort",
                    on_change=EvalState.select_tests_sort,
                    width="14em",
                ),
                rx.select(
                    items=EvalState.tests_scenario_options,
                    value=EvalState.selected_tests_scenario,
                    placeholder="Scenario (All)",
                    on_change=EvalState.select_tests_scenario,
                    width="14em",
                ),
                rx.select(
                    items=EvalState.run_id_options,
                    value=EvalState.selected_run_id,
                    placeholder="Run id (All)",
                    on_change=EvalState.select_run,
                    width="18em",
                ),
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
                            rx.table.column_header_cell("Rating"),
                            rx.table.column_header_cell("Judge comment"),
                        )
                    ),
                    rx.table.body(rx.foreach(EvalState.tests_rows, _row)),
                    variant="surface",
                    style={
                        "width": "100%",
                        "maxWidth": "100%",
                        "tableLayout": "fixed",
                    },
                ),
                type="always",
                scrollbars="vertical",
                style={
                    "maxHeight": "80vh",
                    "width": "100%",
                    "maxWidth": "100%",
                },
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


def _compare_card() -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.heading("Compare runs", size="4"),
            rx.vstack(
                rx.select(
                    items=EvalState.run_options_only,
                    value=EvalState.compare_run_a,
                    placeholder="Run A",
                    on_change=EvalState.set_compare_run_a,
                    width="100%",
                ),
                rx.select(
                    items=EvalState.run_options_only,
                    value=EvalState.compare_run_b,
                    placeholder="Run B",
                    on_change=EvalState.set_compare_run_b,
                    width="100%",
                ),
                spacing="2",
                align="center",
                width="100%",
            ),
            rx.button(
                "Compare",
                on_click=EvalState.compare_runs,
                disabled=~EvalState.can_compare_runs,  # type: ignore[operator]
                loading=EvalState.compare_loading,
                width="100%",
                margin_top="0.5em",
            ),
            rx.cond(
                EvalState.compare_error != "",
                rx.callout(EvalState.compare_error, color_scheme="red", variant="soft"),
                rx.box(),
            ),
            spacing="3",
            width="100%",
        ),
        padding="1em",
        width="100%",
    )


def _compare_dialog() -> rx.Component:
    def _stat_block(label: str, summary: RunSummary) -> rx.Component:
        avg_time = summary["avg_answer_time_sec"]
        median_time = summary["median_answer_time_sec"]
        return rx.card(
            rx.vstack(
                rx.text(label, weight="medium"),
                rx.hstack(
                    rx.badge(
                        rx.text(
                            rx.cond(
                                summary["tests_total"],
                                f"Pass: {summary['passed']}",
                                "No tests",
                            ),
                            size="1",
                        ),
                        color_scheme="green",
                        variant="soft",
                    ),
                    rx.badge(
                        f"Fail: {summary['failed']}",
                        color_scheme="red",
                        variant="soft",
                    ),
                    rx.badge(
                        f"Rating: {summary['avg_rating']}",
                        color_scheme="blue",
                        variant="soft",
                    ),
                    rx.badge(
                        f"Cost: ${summary['cost_total']:.3f}",
                        color_scheme="gray",
                        variant="soft",
                    ),
                    rx.badge(
                        rx.text(
                            f"Time (avg/med): {avg_time} / {median_time}",
                            size="1",
                        ),
                        color_scheme="purple",
                        variant="soft",
                    ),
                    spacing="2",
                    align="center",
                ),
                spacing="2",
            ),
            padding="0.75em",
            width="100%",
        )

    def _config_block(rows: list[dict[str, rx.Var]]) -> rx.Component:
        def _row(row: dict[str, rx.Var]) -> rx.Component:
            return rx.cond(
                row["diff"],
                rx.hstack(
                    rx.text(row["label"], weight="medium", size="1"),
                    rx.spacer(),
                    rx.text(row["value"], size="1"),
                    style={"padding": "0.15em 0.25em", "backgroundColor": "var(--amber-3)"},
                ),
                rx.hstack(
                    rx.text(row["label"], weight="medium", size="1"),
                    rx.spacer(),
                    rx.text(row["value"], size="1"),
                    style={"padding": "0.15em 0.25em"},
                ),
            )

        return rx.card(
            rx.vstack(
                rx.text("Config", weight="medium"),
                rx.foreach(rows, _row),
                spacing="1",
            ),
            padding="0.75em",
            width="100%",
        )

    def _diff_table(title: str, rows: list[dict[str, rx.Var]]) -> rx.Component:
        def _line(row: dict[str, rx.Var]) -> rx.Component:
            left_border = rx.cond(
                row.get("strong", False),
                f"2px solid {row.get('left_color', 'inherit')}",
                "2px solid transparent",
            )
            right_border = rx.cond(
                row.get("strong", False),
                f"2px solid {row.get('right_color', 'inherit')}",
                "2px solid transparent",
            )
            return rx.hstack(
                rx.box(
                    rx.text(
                        row.get("left", ""),
                        size="1",
                        white_space="pre-wrap",
                        weight=rx.cond(row.get("strong", False), "bold", "regular"),
                        color=row.get("left_color", "inherit"),
                    ),
                    style={
                        "fontFamily": "monospace",
                        "padding": "1px 3px",
                        "borderRadius": "4px",
                        "borderLeft": left_border,
                        "width": "100%",
                    },
                ),
                rx.box(
                    rx.text(
                        row.get("right", ""),
                        size="1",
                        white_space="pre-wrap",
                        weight=rx.cond(row.get("strong", False), "bold", "regular"),
                        color=row.get("right_color", "inherit"),
                    ),
                    style={
                        "fontFamily": "monospace",
                        "padding": "1px 3px",
                        "borderRadius": "4px",
                        "borderLeft": right_border,
                        "width": "100%",
                    },
                ),
                spacing="1",
                width="100%",
            )

        return rx.vstack(
            rx.text(title, weight="medium"),
            rx.hstack(
                rx.text(EvalState.compare_run_label_a, weight="medium", size="1"),
                rx.spacer(),
                rx.text(EvalState.compare_run_label_b, weight="medium", size="1"),
                width="100%",
            ),
            rx.scroll_area(
                rx.vstack(
                    rx.foreach(rows, _line),
                    spacing="1",
                    width="100%",
                ),
                type="always",
                scrollbars="vertical",
                style={"maxHeight": "22vh", "padding": "1px"},
            ),
            spacing="1",
            width="100%",
        )

    def _result_row(row: dict[str, rx.Var]) -> rx.Component:
        def _badge_color(status: str | None) -> rx.Var:
            """
            green for pass, red for fail, grey else
            """
            return rx.cond(
                status == "pass",
                "green",
                rx.cond(status == "fail", "red", "gray"),
            )

        def _answer_block(text: str, tool_calls: str) -> rx.Component:
            return rx.vstack(
                rx.text(text, size="1", white_space="pre-wrap", style={"wordBreak": "break-word"}),
                rx.cond(
                    tool_calls != "",
                    rx.accordion.root(
                        rx.accordion.item(
                            rx.accordion.trigger("Tool calls", style={"fontSize": "12px"}),
                            rx.accordion.content(
                                rx.text(
                                    tool_calls,
                                    size="1",
                                    color="gray",
                                    white_space="pre-wrap",
                                    style={"wordBreak": "break-word"},
                                )
                            ),
                            value="tool-calls",
                        ),
                        collapsible=True,
                        type="single",
                        default_value="",
                        variant="ghost",
                        style={"width": "100%"},
                    ),
                    rx.box(),
                ),
                spacing="1",
                align="start",
                width="100%",
            )

        return rx.table.row(
            rx.table.cell(
                rx.vstack(
                    rx.text(row["question"]),
                    rx.text(
                        row["golden_answer"],
                        size="1",
                        color="gray",
                        white_space="pre-wrap",
                        style={"wordBreak": "break-word"},
                    ),
                    spacing="1",
                    align="start",
                    width="100%",
                )
            ),
            rx.table.cell(
                rx.vstack(
                    rx.hstack(
                        rx.badge(
                            row["status_a"],
                            color_scheme=_badge_color(row["status_a"]),  # type: ignore[arg-type]
                            variant="soft",
                        ),
                        rx.text(f"Rating: {row['rating_a']}", size="1"),
                        align="center",
                    ),
                    _answer_block(row["answer_a"], row["tool_calls_a"]),  # type: ignore[arg-type]
                    rx.text(row["comment_a"], size="1", color="gray"),
                    spacing="1",
                    align="start",
                )
            ),
            rx.table.cell(
                rx.vstack(
                    rx.hstack(
                        rx.badge(
                            row["status_b"],
                            color_scheme=_badge_color(row["status_b"]),  # type: ignore[arg-type]
                            variant="soft",
                        ),
                        rx.text(f"Rating: {row['rating_b']}", size="1"),
                        align="center",
                    ),
                    _answer_block(row["answer_b"], row["tool_calls_b"]),  # type: ignore[arg-type]
                    rx.text(row["comment_b"], size="1", color="gray"),
                    spacing="1",
                    align="start",
                )
            ),
        )

    return rx.dialog.root(
        rx.dialog.content(
            rx.dialog.title("Run comparison"),
            rx.vstack(
                rx.hstack(
                    _stat_block(EvalState.compare_run_label_a, EvalState.compare_summary_a),  # type: ignore[arg-type]
                    _stat_block(EvalState.compare_run_label_b, EvalState.compare_summary_b),  # type: ignore[arg-type]
                    spacing="3",
                    width="100%",
                ),
                rx.hstack(
                    _config_block(EvalState.compare_config_a_rows),  # type: ignore[arg-type]
                    _config_block(EvalState.compare_config_b_rows),  # type: ignore[arg-type]
                    spacing="3",
                    width="100%",
                ),
                rx.cond(
                    EvalState.compare_diff_keys != [],
                    rx.card(
                        rx.vstack(
                            rx.hstack(
                                rx.text("Differences:", weight="medium", size="1"),
                                rx.box(rx.foreach(EvalState.compare_diff_keys, lambda k: rx.badge(k, variant="soft"))),
                                spacing="3",
                                align="center",
                            ),
                            rx.accordion.root(
                                rx.accordion.item(
                                    rx.accordion.trigger("Judge prompt diff"),
                                    rx.accordion.content(
                                        rx.vstack(
                                            rx.checkbox(
                                                "Show only changes",
                                                default_checked=True,
                                                checked=EvalState.compare_judge_prompt_compact,
                                                on_change=EvalState.set_compare_judge_prompt_compact,
                                                size="2",
                                            ),
                                            _diff_table("Judge prompt diff", EvalState.compare_prompt_rows_view),  # type: ignore[arg-type]
                                            spacing="1",
                                        )
                                    ),
                                    value="prompt-diff-block",
                                ),
                                rx.accordion.item(
                                    rx.accordion.trigger("Data model diff"),
                                    rx.accordion.content(
                                        rx.vstack(
                                            rx.checkbox(
                                                "Show only changes",
                                                default_checked=True,
                                                checked=EvalState.compare_dm_compact,
                                                on_change=EvalState.set_compare_dm_compact,
                                                size="2",
                                            ),
                                            _diff_table("Data model diff", EvalState.compare_dm_rows_view),  # type: ignore[arg-type]
                                            spacing="1",
                                        )
                                    ),
                                    value="dm-diff-block",
                                ),
                                type="multiple",
                                collapsible=True,
                                variant="outline",
                                width="100%",
                            ),
                            spacing="2",
                            width="100%",
                        ),
                        variant="surface",
                        width="100%",
                    ),
                    rx.box(),
                ),
                rx.cond(
                    EvalState.compare_loading,
                    rx.center(rx.spinner(size="3"), height="200px"),
                    rx.scroll_area(
                        rx.table.root(
                            rx.table.header(
                                rx.table.row(
                                    rx.table.column_header_cell("Question", style={"width": "25%"}),
                                    rx.table.column_header_cell(
                                        EvalState.compare_run_label_a, style={"width": "37.5%"}
                                    ),
                                    rx.table.column_header_cell(
                                        EvalState.compare_run_label_b, style={"width": "37.5%"}
                                    ),
                                )
                            ),
                            rx.table.body(rx.foreach(EvalState.compare_rows, _result_row)),
                            variant="surface",
                            style={"width": "100%", "tableLayout": "fixed"},
                        ),
                        type="always",
                        scrollbars="vertical",
                        style={"maxHeight": "60vh"},
                    ),
                ),
                rx.dialog.close(rx.button("Close", variant="soft")),
                spacing="3",
                width="100%",
            ),
            style={"maxWidth": "92vw"},
        ),
        open=EvalState.compare_dialog_open,
        on_open_change=EvalState.set_compare_dialog_open,
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
            rx.callout(
                rx.vstack(
                    rx.text(EvalState.status_message, weight="medium"),
                    rx.cond(
                        EvalState.current_question_progress != "",
                        rx.text(EvalState.current_question_progress, size="1", color="gray"),  # type: ignore[arg-type]
                        rx.box(),
                    ),
                    spacing="1",
                ),
                color_scheme="green",
                variant="soft",
                width="100%",
            ),
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
                rx.vstack(
                    _questions_card(),
                    _tests_card(),
                ),
                rx.vstack(
                    _judge_card(),
                    _pipeline_card(),
                    _selection_and_actions(),
                    _status_messages(),
                    _compare_card(),
                    spacing="4",
                    width="100%",
                ),
                columns="2",
                spacing="4",
                width="100%",
                style={"gridTemplateColumns": "3fr 1fr", "height": "calc(100vh - 200px)", "minHeight": "700px"},
            ),
            spacing="4",
            width="100%",
        ),
        _compare_dialog(),
        _judge_prompt_dialog(),
        _data_model_dialog(),
        align="start",
        spacing="2",
        padding="1.5em",
    )
