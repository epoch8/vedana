from uuid import UUID

from uuid_extensions import uuid7 as uuid7_impl

# litellm.callbacks = ["otel"]


def uuid7() -> UUID:
    return uuid7_impl(as_type="UUID")  # type: ignore
