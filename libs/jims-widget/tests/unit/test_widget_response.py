"""Tests for assistant message + button formatting."""

from types import SimpleNamespace

import pytest
from jims_widget.server import _buttons_to_widget_html, _normalize_button_rows, _widget_response_from_outgoing


def test_normalize_button_rows_flat_row() -> None:
    flat = [{"text": "A", "id": "a"}]
    assert _normalize_button_rows(flat) == [flat]


def test_normalize_button_rows_nested() -> None:
    rows = [[{"text": "A", "id": "a"}], [{"text": "B", "id": "b"}]]
    assert _normalize_button_rows(rows) == rows


def test_buttons_to_widget_html_callback() -> None:
    html = _buttons_to_widget_html([[{"text": "Позвать оператора", "id": "operator_handoff"}]])
    assert "jims-widget-callback-btn" in html
    assert "btn:operator_handoff" in html
    assert "Позвать оператора" in html


def test_buttons_to_widget_html_prefixed_id() -> None:
    html = _buttons_to_widget_html([[{"text": "X", "id": "btn:already"}]])
    assert 'data-jims-callback="btn:already"' in html


@pytest.mark.parametrize(
    ("event_type", "event_data", "expect_text_sub", "expect_html"),
    [
        (
            "comm.assistant_message",
            {"role": "assistant", "content": "Hello"},
            "Hello",
            False,
        ),
        (
            "comm.assistant_buttons",
            {
                "role": "assistant",
                "content": "Low confidence?",
                "buttons": [[{"text": "Op", "id": "operator_handoff"}]],
            },
            "Low confidence?",
            True,
        ),
    ],
)
def test_widget_response_from_outgoing(
    event_type: str,
    event_data: dict,
    expect_text_sub: str,
    expect_html: bool,
) -> None:
    ev = SimpleNamespace(event_type=event_type, event_data=event_data)
    out = _widget_response_from_outgoing([ev])
    if isinstance(out, list):
        assert len(out) == 2
        text_blob = out[0]["text"]
        html_blob = out[1]["html"]
    else:
        text_blob = out["text"]
        html_blob = out.get("html") or ""
    assert expect_text_sub in text_blob
    if expect_html:
        assert "jims-widget-callback-btn" in html_blob
    else:
        assert not html_blob
