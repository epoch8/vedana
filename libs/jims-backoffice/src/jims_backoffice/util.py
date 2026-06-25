import csv
import io
from datetime import datetime
from typing import Any, Iterable

import openpyxl
import orjson as json


def datetime_to_age(created_at: datetime, compact: bool = True) -> str:
    """Convert a datetime to a human-readable age string.

    For datetimes less than 7 days old, returns relative time.
    For older datetimes, returns absolute date format (e.g., "2025 Aug 24 14:30").

    Args:
        created_at: The datetime to convert
        compact: If True, uses compact format (e.g., "2h", "3d14h").
                If False, uses verbose format (e.g., "2 hours", "3 days 14 hours").
    """
    now = datetime.now()
    time_diff = now - created_at

    if time_diff.days < 7:
        if time_diff.days > 0:
            hours = time_diff.seconds // 3600
            if hours > 0:
                if compact:
                    return f"{time_diff.days}d{hours}h"
                else:
                    return f"{time_diff.days} days {hours} hours"
            else:
                if compact:
                    return f"{time_diff.days}d"
                else:
                    return f"{time_diff.days} days"
        elif time_diff.seconds >= 3600:
            hours = time_diff.seconds // 3600
            if compact:
                return f"{hours}h"
            else:
                return f"{hours} hours"
        elif time_diff.seconds >= 60:
            minutes = time_diff.seconds // 60
            if compact:
                return f"{minutes}m"
            else:
                return f"{minutes} minutes"
        else:
            if compact:
                return "1m"
            else:
                return "1 minute"
    else:
        return created_at.strftime("%Y %b %d %H:%M")


def safe_render_value(v: Any) -> str:
    try:
        if isinstance(v, (dict, list)):
            s = json.dumps(v).decode()
        elif v is None:
            s = ""
        else:
            s = str(v)
    except Exception:
        try:
            s = repr(v)
        except Exception:
            s = "<error rendering value>"
    return s


_XLSX_MAX_CELL = 32767  # Excel per-cell character limit


def xlsx_bytes(rows: Iterable[dict[str, Any]], columns: list[str]) -> bytes:
    """Serialize rows into an Excel (.xlsx) file returned as bytes.

    Values are coerced via `safe_render_value`. Strings longer than Excel's
    32 767-character cell limit are truncated with a trailing ellipsis.
    """
    wb = openpyxl.Workbook(write_only=True)
    ws = wb.create_sheet()

    ws.append(columns)
    for row in rows:
        cells = []
        for col in columns:
            val = safe_render_value(row.get(col, ""))
            if len(val) > _XLSX_MAX_CELL:
                val = val[: _XLSX_MAX_CELL - 1] + "…"
            cells.append(val)
        ws.append(cells)

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def csv_bytes(rows: Iterable[dict[str, Any]], columns: list[str]) -> bytes:
    """Serialize a sequence of dict rows into UTF-8 encoded CSV bytes.

    Missing columns default to "". Non-string values are coerced via
    `safe_render_value` so nested dicts/lists become compact JSON.
    """
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        flat = {col: safe_render_value(row.get(col, "")) for col in columns}
        writer.writerow(flat)
    return buf.getvalue().encode("utf-8")
