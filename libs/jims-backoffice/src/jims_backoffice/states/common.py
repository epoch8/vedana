import os

import reflex as rx

EVAL_ENABLED = bool(os.environ.get("GRIST_TEST_SET_DOC_ID"))
DEBUG_MODE = (
    os.environ.get("JIMS_BACKOFFICE_DEBUG", "").lower() in ("true", "1")
    or os.environ.get("DEBUG", "").lower() in ("true", "1")
)


class AppVersionState(rx.State):
    version: str = f"`{os.environ.get('VERSION', 'unspecified_version')}`"  # md-formatted
    eval_enabled: bool = EVAL_ENABLED
    debug_mode: bool = DEBUG_MODE
