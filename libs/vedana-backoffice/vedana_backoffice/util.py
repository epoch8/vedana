from datetime import datetime, timezone


def _datetime_to_age(created_at: datetime) -> str:
    from datetime import datetime as dt
    from datetime import timezone

    now = dt.now(timezone.utc)
    created_at_dt = created_at
    if created_at_dt.tzinfo is None:
        created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)

    diff = now - created_at_dt
    if diff.days < 7:
        if diff.days > 0:
            hours = diff.seconds // 3600
            return f"{diff.days}d{hours}h" if hours > 0 else f"{diff.days}d"
        if diff.seconds >= 3600:
            return f"{diff.seconds // 3600}h"
        if diff.seconds >= 60:
            return f"{diff.seconds // 60}m"
        return "1m"
    return created_at_dt.strftime("%Y %b %d %H:%M")


def datetime_to_age(created_at: datetime, compact: bool = True) -> str:
    """Convert a datetime to a human-readable age string.

    For datetimes less than 7 days old, returns relative time.
    For older datetimes, returns absolute date format (e.g., "2025 Aug 24 14:30").

    Args:
        created_at: The datetime to convert
        compact: If True, uses compact format (e.g., "2h", "3d14h").
                If False, uses verbose format (e.g., "2 hours", "3 days 14 hours").
    """
    now = datetime.now(timezone.utc)
    created_at_dt = created_at
    if created_at_dt.tzinfo is None:
        created_at_dt = created_at_dt.replace(tzinfo=timezone.utc)

    time_diff = now - created_at_dt

    if time_diff.days < 7:
        # Human readable relative time
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
        # Absolute date format
        return created_at_dt.strftime("%Y %b %d %H:%M")
