from datetime import datetime
from typing import Any

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
                return f"{time_diff.days}d{hours}h" if compact else f"{time_diff.days} days {hours} hours"
            else:
                return f"{time_diff.days}d" if compact else f"{time_diff.days} days"
        elif time_diff.seconds >= 3600:
            hours = time_diff.seconds // 3600
            return f"{hours}h" if compact else f"{hours} hours"
        elif time_diff.seconds >= 60:
            minutes = time_diff.seconds // 60
            return f"{minutes}m" if compact else f"{minutes} minutes"
        else:
            return "1m" if compact else "1 minute"
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
