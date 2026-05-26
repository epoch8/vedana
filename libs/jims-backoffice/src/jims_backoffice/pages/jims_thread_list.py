from typing import Callable

import reflex as rx

from jims_backoffice.components.ui_chat import render_message_bubble
from jims_backoffice.states.jims import ThreadListState, ThreadViewState
from jims_backoffice.ui import app_header

def jims_thread_list_page(
    *,
    header: Callable[[], rx.Component] | None = None,
) -> rx.Component:
    def _badge_style(bg: str, fg: str) -> dict[str, str]:
        return {
            "backgroundColor": bg,
            "color": fg,
        }

    def review_badge(value: str) -> rx.Component:  # type: ignore[valid-type]
        return rx.cond(
            value == "Pending",
            rx.badge("Pending", variant="soft", size="1", style=_badge_style("var(--amber-4)", "var(--amber-11)")),
            rx.cond(
                value == "Complete",
                rx.badge("Complete", variant="soft", size="1", style=_badge_style("var(--green-4)", "var(--green-11)")),
                rx.badge(value, variant="soft", size="1", style=_badge_style("var(--gray-4)", "var(--gray-12)")),
            ),
        )

    def priority_badge(value: str) -> rx.Component:  # type: ignore[valid-type]
        return rx.cond(
            value == "High",
            rx.badge("High", variant="soft", size="1", style=_badge_style("var(--tomato-4)", "var(--tomato-11)")),
            rx.cond(
                value == "Medium",
                rx.badge("Medium", variant="soft", size="1", style=_badge_style("var(--amber-4)", "var(--amber-11)")),
                rx.badge(value, variant="soft", size="1", style=_badge_style("var(--gray-4)", "var(--gray-12)")),
            ),
        )

    bulk_export_dialog = rx.dialog.root(
        rx.dialog.content(
            rx.vstack(
                rx.dialog.title("Export thread_events"),
                rx.text(
                    ThreadListState.bulk_export_confirm_message,
                    size="2",
                ),
                rx.hstack(
                    rx.button(
                        "Cancel",
                        variant="soft",
                        on_click=ThreadListState.close_bulk_export_dialog,
                    ),
                    rx.button(
                        "Export",
                        on_click=ThreadListState.confirm_bulk_export,
                        disabled=ThreadListState.total_threads == 0,
                    ),
                    spacing="2",
                    justify="end",
                    width="100%",
                ),
                spacing="3",
                width="100%",
            ),
        ),
        open=ThreadListState.bulk_export_dialog_open,
        on_open_change=ThreadListState.handle_bulk_export_dialog_open_change,
    )

    filters = rx.hstack(
        rx.hstack(
            rx.hstack(
                rx.input(
                    value=ThreadListState.from_date,
                    type="date",
                    on_change=ThreadListState.set_from_date,
                ),
                rx.text("-"),
                rx.input(
                    value=ThreadListState.to_date,
                    type="date",
                    on_change=ThreadListState.set_to_date,
                ),
                align="center",
                spacing="1",
            ),
            rx.select(
                items=["All", "Pending", "Complete"],
                placeholder="Review: All",
                on_change=ThreadListState.set_review_filter,
                width="180px",
            ),
            rx.dialog.root(
                rx.dialog.trigger(
                    rx.button(
                        rx.cond(
                            ThreadListState.selected_tags.length() > 0,  # type: ignore[attr-defined]
                            "Tags: " + ThreadListState.selected_tags.join(", "),  # type: ignore[attr-defined]
                            "Tags: All",
                        ),
                        variant="soft",
                        color_scheme="gray",
                    )
                ),
                rx.dialog.content(
                    rx.vstack(
                        rx.hstack(
                            rx.dialog.title("Filter by Tags"),
                            rx.dialog.close(rx.button("Close", variant="ghost", size="1")),
                            justify="between",
                            align="center",
                            width="100%",
                        ),
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(
                                    ThreadListState.available_tags,
                                    lambda t: rx.checkbox(
                                        t,
                                        checked=ThreadListState.selected_tags.contains(t),  # type: ignore[attr-defined]
                                        on_change=lambda v, tag=t: ThreadListState.toggle_tag_filter(tag=tag, value=v),  # type: ignore[operator]
                                    ),
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            type="always",
                            scrollbars="vertical",
                        ),
                        rx.hstack(
                            rx.dialog.close(
                                rx.button(
                                    "Apply",
                                    on_click=[ThreadListState.reset_pagination, ThreadListState.get_data],
                                    size="1",
                                )
                            ),
                            rx.button(
                                "Clear",
                                variant="soft",
                                on_click=[ThreadListState.clear_tag_filter, ThreadListState.get_data],
                                size="1",
                            ),
                            spacing="2",
                            justify="end",
                            width="100%",
                        ),
                        spacing="3",
                    )
                ),
            ),
            rx.hstack(
                rx.select(
                    items=["Date", "Priority"],
                    on_change=ThreadListState.set_sort_by,
                    placeholder="Sort By: Date",
                    width="160px",
                ),
                rx.cond(
                    ThreadListState.sort_reverse,
                    rx.icon(
                        "arrow-down-1-0",
                        size=28,
                        stroke_width=1.5,
                        cursor="pointer",
                        flex_shrink="0",
                        on_click=ThreadListState.toggle_sort,
                    ),  # type: ignore
                    rx.icon(
                        "arrow-down-0-1",
                        size=28,
                        stroke_width=1.5,
                        cursor="pointer",
                        flex_shrink="0",
                        on_click=ThreadListState.toggle_sort,
                    ),  # type: ignore
                ),
                spacing="0",
            ),
            rx.select(
                items=ThreadListState.available_interfaces,
                placeholder="Interface: All",
                on_change=ThreadListState.set_selected_interface,
                width="200px",
            ),
            rx.input(
                placeholder="Search thread_id",
                value=ThreadListState.search_text,
                on_change=ThreadListState.set_search_text,
                width="280px",
            ),
        ),
        rx.hstack(
            rx.button("Search", on_click=[ThreadListState.reset_pagination, ThreadListState.get_data]),
            rx.button(
                "Clear",
                variant="soft",
                on_click=[ThreadListState.clear_filters, ThreadListState.get_data],
            ),
            rx.button(
                rx.icon("download"),
                variant="soft",
                on_click=ThreadListState.open_bulk_export_dialog,
            ),
            bulk_export_dialog,
            spacing="2",
        ),
        justify="between",
        width="100%",
        align="center",
        wrap="wrap",
    )

    table = rx.table.root(
        rx.table.header(
            rx.table.row(
                rx.table.column_header_cell("Thread ID"),
                rx.table.column_header_cell("Created"),
                rx.table.column_header_cell("Age"),
                rx.table.column_header_cell("Interface"),
                rx.table.column_header_cell("Tags"),
                rx.table.column_header_cell("Review"),
                rx.table.column_header_cell("Priority"),
            ),
        ),
        rx.table.body(
            rx.foreach(
                ThreadListState.threads,
                lambda t: rx.table.row(
                    rx.table.cell(
                        rx.button(
                            t.thread_id,
                            variant="ghost",
                            color_scheme="gray",
                            size="1",
                            on_click=ThreadViewState.select_thread(thread_id=t.thread_id),  # type: ignore[operator]
                        )
                    ),  # type: ignore[call-arg,func-returns-value]
                    rx.table.cell(t.created_at),
                    rx.table.cell(t.thread_age),
                    rx.table.cell(t.interface),
                    rx.table.cell(
                        rx.hstack(
                            rx.cond(t.tag1 != "", rx.badge(t.tag1, variant="soft", size="1", color_scheme="gray")),
                            rx.cond(t.tag2 != "", rx.badge(t.tag2, variant="soft", size="1", color_scheme="gray")),
                            rx.cond(t.tag3 != "", rx.badge(t.tag3, variant="soft", size="1", color_scheme="gray")),
                            spacing="1",
                        )
                    ),
                    rx.table.cell(review_badge(t.review_status)),
                    rx.table.cell(priority_badge(t.priority)),
                    style=rx.cond(
                        t.thread_id == ThreadViewState.selected_thread_id, {"backgroundColor": "var(--accent-3)"}, {}
                    ),
                ),
            ),
        ),
    )

    pagination_controls = rx.hstack(
        rx.text(ThreadListState.rows_display, size="2", color="gray"),
        rx.spacer(),
        rx.hstack(
            rx.button(
                "⏮",
                variant="soft",
                size="1",
                on_click=ThreadListState.first_page,
                disabled=~ThreadListState.has_prev_page,
            ),  # type: ignore[operator]
            rx.button(
                "← Prev",
                variant="soft",
                size="1",
                on_click=ThreadListState.prev_page,
                disabled=~ThreadListState.has_prev_page,
            ),  # type: ignore[operator]
            rx.text(ThreadListState.page_display, size="2", style={"minWidth": "110px", "textAlign": "center"}),
            rx.button(
                "Next →",
                variant="soft",
                size="1",
                on_click=ThreadListState.next_page,
                disabled=~ThreadListState.has_next_page,
            ),  # type: ignore[operator]
            rx.button(
                "⏭",
                variant="soft",
                size="1",
                on_click=ThreadListState.last_page,
                disabled=~ThreadListState.has_next_page,
            ),  # type: ignore[operator]
            spacing="2",
            align="center",
        ),
        width="100%",
        align="center",
        padding_top="0.5em",
    )

    def _render_event_as_msg(ev):  # type: ignore[valid-type]
        msg = {
            "id": ev.event_id,
            "content": ev.content,
            "created_at": ev.created_at_str,
            "is_assistant": ev.role == "assistant",
            "tag_label": ev.event_type,
            "tags": ev.visible_tags,
            "comments": ev.feedback_comments,
            "has_tech": ev.has_technical_info,
            "has_models": ev.has_models,
            "has_vts": ev.has_vts,
            "has_cypher": ev.has_cypher,
            "models_str": ev.models_str,
            "vts_str": ev.vts_str,
            "cypher_str": ev.cypher_str,
            "show_details": ThreadViewState.expanded_event_id == ev.event_id,
            "event_data_str": ev.event_data_str,
            "generic_meta": ev.generic_meta,
        }

        tag_dialog = rx.dialog.root(
            rx.dialog.content(
                rx.vstack(
                    rx.hstack(
                        rx.dialog.title("Add Tags"),
                        rx.dialog.close(
                            rx.button("Close", variant="ghost", size="1", on_click=ThreadViewState.close_tag_dialog)
                        ),
                        justify="between",
                        align="center",
                        width="100%",
                    ),
                    rx.hstack(
                        rx.input(
                            placeholder="Add new tag...",
                            value=ThreadViewState.new_tag_text_for_event.get(ev.event_id, ""),
                            on_change=lambda v: ThreadViewState.set_new_tag_text_for_event(v, event_id=ev.event_id),
                            width="100%",
                        ),
                        rx.button(
                            "Add",
                            size="1",
                            on_click=ThreadViewState.add_new_tag_to_available(event_id=ev.event_id),
                        ),
                        spacing="2",
                        width="100%",
                    ),
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(
                                ThreadViewState.available_tags,
                                lambda t: rx.checkbox(
                                    t,
                                    checked=ThreadViewState.selected_tags_for_event.get(ev.event_id, []).contains(t),  # type: ignore[attr-defined]
                                    on_change=lambda v, tag=t: ThreadViewState.toggle_tag_selection_for_event(
                                        tag=tag, event_id=ev.event_id, checked=v
                                    ),  # type: ignore[operator]
                                ),
                            ),
                            spacing="2",
                            width="100%",
                        ),
                        type="always",
                        scrollbars="vertical",
                        style={"width": "100%", "padding": "0"},
                    ),
                    rx.hstack(
                        rx.button(
                            "Apply",
                            size="1",
                            on_click=ThreadViewState.apply_tags_to_event(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                        ),
                        spacing="2",
                        justify="end",
                        width="100%",
                    ),
                    spacing="3",
                ),
            ),
            open=ThreadViewState.tag_dialog_open_for_event == ev.event_id,
            on_open_change=ThreadViewState.handle_tag_dialog_open_change,  # type: ignore[operator]
        )

        action_line = rx.cond(
            (ev.role == "assistant") & ev.content != "",
            rx.vstack(
                rx.text_area(
                    placeholder="Add note...",
                    value=ThreadViewState.note_text_by_event.get(ev.event_id, ""),
                    on_change=lambda v: ThreadViewState.set_note_text_for(v, event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                    width="100%",
                    size="1",
                    class_name="jims-note-textarea",
                ),
                rx.hstack(
                    rx.button(
                        "Add tag",
                        size="1",
                        on_click=ThreadViewState.open_tag_dialog(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                    ),
                    tag_dialog,
                    rx.select(
                        items=["Low", "Medium", "High"],
                        value=ThreadViewState.note_severity_by_event.get(ev.event_id, "Low"),
                        on_change=lambda v: ThreadViewState.set_note_severity_for(v, event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                        size="1",
                    ),
                    rx.button(
                        "Add note",
                        size="1",
                        on_click=ThreadViewState.submit_note_for(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
                    ),
                    spacing="2",
                    wrap="wrap",
                    width="100%",
                    align="center",
                ),
                spacing="2",
                width="100%",
                align="stretch",
            ),
            rx.box(),
        )

        comments_component = rx.vstack(
            rx.foreach(
                ev.feedback_comments,
                lambda c: rx.card(
                    rx.vstack(
                        rx.hstack(
                            rx.cond(
                                c["severity"] == "High",
                                rx.badge("High", variant="soft", size="1", color_scheme="tomato"),
                                rx.cond(
                                    c["severity"] == "Medium",
                                    rx.badge("Medium", variant="soft", size="1", color_scheme="amber"),
                                    rx.badge("Low", variant="soft", size="1", color_scheme="gray"),
                                ),
                            ),
                            rx.text(c["created_at"], size="1", color="gray"),
                            rx.spacer(),
                            rx.cond(
                                c.get("status", "open") == "resolved",
                                rx.badge("Resolved", variant="soft", size="1", color_scheme="green"),
                                rx.cond(
                                    c.get("status", "open") == "closed",
                                    rx.badge("Ignored", variant="soft", size="1", color_scheme="gray"),
                                    rx.box(),
                                ),
                            ),
                            spacing="2",
                            align="center",
                            width="100%",
                        ),
                        rx.text(
                            c["note"],
                            style={
                                "whiteSpace": "pre-wrap",
                                "wordBreak": "break-word",
                            },
                        ),
                        rx.hstack(
                            rx.button(
                                "✔",
                                variant="soft",
                                size="1",
                                color_scheme="green",
                                disabled=c.get("status", "open") != "open",
                                on_click=ThreadViewState.mark_comment_resolved(comment_id=c["id"]),  # type: ignore[call-arg,func-returns-value]
                            ),
                            rx.button(
                                "✖",
                                variant="soft",
                                size="1",
                                color_scheme="gray",
                                disabled=c.get("status", "open") != "open",
                                on_click=ThreadViewState.mark_comment_closed(comment_id=c["id"]),  # type: ignore[call-arg,func-returns-value]
                            ),
                            rx.cond(
                                c.get("status", "open") != "open",
                                rx.tooltip(
                                    rx.button(
                                        "↺",
                                        variant="soft",
                                        size="1",
                                        color_scheme="amber",
                                        on_click=ThreadViewState.reopen_comment(comment_id=c["id"]),  # type: ignore[call-arg,func-returns-value]
                                    ),
                                    content="Reopen comment",
                                ),
                                rx.box(),
                            ),
                            rx.spacer(),
                            rx.button(
                                "🗑",
                                variant="soft",
                                size="1",
                                color_scheme="tomato",
                                on_click=ThreadViewState.delete_comment(comment_id=c["id"]),  # type: ignore[call-arg,func-returns-value]
                            ),
                            spacing="2",
                            width="100%",
                        ),
                        spacing="1",
                        width="100%",
                    ),
                    padding="0.5em",
                    variant="surface",
                ),
            ),
            spacing="2",
            width="100%",
        )

        extras = rx.vstack(
            comments_component,
            action_line,
            spacing="2",
            width="100%",
            align="stretch",
        )

        def _tag_badge(tag: str):  # type: ignore[valid-type]
            return rx.badge(
                rx.hstack(
                    rx.text(tag),
                    rx.button(
                        "×",
                        variant="ghost",
                        size="1",
                        color_scheme="gray",
                        on_click=ThreadViewState.remove_tag(event_id=ev.event_id, tag=tag),  # type: ignore[operator, call-arg,func-returns-value]
                    ),
                    spacing="1",
                ),
                variant="soft",
                size="1",
                color_scheme="gray",
            )

        tags_component = rx.hstack(
            rx.foreach(ev.visible_tags, _tag_badge),
            spacing="1",
        )

        return render_message_bubble(
            msg,
            on_toggle_details=ThreadViewState.toggle_details(event_id=ev.event_id),  # type: ignore[call-arg,func-returns-value]
            extras=extras,
            corner_tags_component=tags_component,
        )

    filters_box = rx.box(
        filters,
        margin_bottom="1em",
    )

    left_panel = rx.box(
        rx.vstack(
            rx.cond(
                ThreadListState.threads_refreshing,
                rx.center("Loading threads..."),
                rx.scroll_area(
                    table,
                    type="always",
                    scrollbars="vertical",
                    style={"height": "100%"},
                ),
            ),
            pagination_controls,
            width="100%",
            spacing="2",
            height="100%",
        ),
        flex="1",
        min_height="0",
        style={"height": "100%", "overflow": "hidden"},
    )

    right_panel_header = rx.hstack(
        rx.text(ThreadViewState.selected_thread_id, size="2", weight="medium"),
        rx.spacer(),
        rx.button(
            rx.icon("download"),
            size="1",
            variant="ghost",
            on_click=ThreadViewState.export_thread_csv,
        ),
        align="center",
        width="100%",
        padding_bottom="0.5em",
    )

    right_panel = rx.cond(
        ThreadViewState.selected_thread_id == "",
        rx.center(rx.text("Select a thread to view conversation"), style={"height": "100%"}),
        rx.flex(
            right_panel_header,
            rx.box(
                rx.scroll_area(
                    rx.vstack(
                        rx.cond(
                            ThreadViewState.has_more_history,
                            rx.center(
                                rx.button(
                                    "Load more",
                                    variant="soft",
                                    size="1",
                                    on_click=ThreadViewState.load_more_history,
                                ),
                                width="100%",
                                padding_top="0.5em",
                            ),
                            rx.box(),
                        ),
                        rx.foreach(ThreadViewState.events, _render_event_as_msg),
                        spacing="3",
                        width="100%",
                        padding_bottom="1em",
                        padding_x="0.25em",
                        style={
                            "maxWidth": "100%",
                            "minWidth": "0",
                        },
                    ),
                    type="always",
                    scrollbars="vertical",
                    class_name="jims-thread-scroll",
                    style={"height": "100%", "width": "100%"},
                ),
                flex="1",
                min_height="0",
                height="0",
                width="100%",
                overflow="hidden",
            ),
            direction="column",
            width="100%",
            height="100%",
            min_height="0",
            flex="1",
        ),
    )

    header_component = (header or app_header)()

    return rx.vstack(
        header_component,
        filters_box,
        rx.box(
            rx.grid(
                left_panel,
                rx.box(
                    right_panel,
                    min_height="0",
                    height="100%",
                    overflow="hidden",
                    style={"minWidth": "0"},
                ),
                columns="2",
                spacing="4",
                sm_columns="1",
                width="100%",
                style={"height": "100%", "minHeight": "0", "alignItems": "stretch"},
            ),
            flex="1",
            min_height="0",
            width="100%",
        ),
        spacing="4",
        height="100vh",
        overflow="hidden",
        width="100%",
    )
