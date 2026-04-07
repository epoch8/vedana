import contextlib
import datetime
import os
import sys
import traceback

import orjson
from loguru import logger


def get_current_datetime() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)


def json_serializer(message):
    record = message.record
    simplified = {
        "level": record["level"].name,
        "message": record["message"],
        "timestamp": record["time"].isoformat(),
        **record["extra"],
    }
    if record.get("exception") is not None:
        simplified["exception"] = traceback.format_exception(
            record["exception"].type,
            value=record["exception"].value,
            tb=record["exception"].traceback,
        )
    serialized = orjson.dumps(simplified, option=orjson.OPT_APPEND_NEWLINE)
    sys.stdout.buffer.write(serialized)


def setup_logging():
    if os.getenv("HUMAN_LOGS", "") == "":
        with contextlib.suppress(ValueError):
            logger.remove(0)

        logger.add(
            json_serializer,
            level="INFO",
            format="{level}: {time} [{name}] {message}",
        )
