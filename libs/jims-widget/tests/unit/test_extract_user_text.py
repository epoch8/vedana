"""Unit tests for DeepChat payload parsing."""

import pytest
from jims_widget.server import _extract_user_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # DeepChat JSON with messages array
        ('{"messages":[{"role":"user","text":"hello"}]}', "hello"),
        ('{"messages":[{"role":"user","text":"  hi  "}]}', "hi"),
        ('{"messages":[{"role":"assistant","text":"x"},{"role":"user","text":"last"}]}', "last"),
        # Double-encoded JSON too
        ('"{\\"messages\\":[{\\"role\\":\\"user\\",\\"text\\":\\"test\\"}]}"', "test"),
        ('"{\\"messages\\":[{\\"role\\":\\"user\\",\\"text\\":\\"hello world\\"}]}"', "hello world"),
        # Plain string
        ("hello", "hello"),
        ("  hi  ", "hi"),
        # JSON-encoded string
        ('"hello"', "hello"),
        # Top-level keys
        ('{"text":"foo"}', "foo"),
        ('{"message":"bar"}', "bar"),
        ('{"content":"baz"}', "baz"),
        # Empty / whitespace
        ("", None),
        ("   ", None),
        # No valid message in dict -> fallback to raw
        ("{}", "{}"),
        ('{"messages":[]}', '{"messages":[]}'),
        (
            '{"messages":[{"role":"user","text":""}]}',
            '{"messages":[{"role":"user","text":""}]}',
        ),  # no valid text → fallback
        ('{"messages":[{"role":"user","text":"  "}]}', None),
        # Invalid JSON -> fallback to raw
        ("not json", "not json"),
        ("  raw  ", "raw"),
    ],
)
def test_extract_user_text(raw: str, expected: str | None) -> None:
    assert _extract_user_text(raw) == expected
