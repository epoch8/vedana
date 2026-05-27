"""Vedana JIMS thread list page — thin wrapper around the jims-backoffice page.

This module is kept under the historical ``jims_thread_list_page`` name so the
old import paths in :mod:`vedana_backoffice.vedana_backoffice` continue to work.
"""

from __future__ import annotations

import jims_backoffice
import reflex as rx

from vedana_backoffice.ui import app_header

ThreadListState = jims_backoffice.ThreadListState
ThreadViewState = jims_backoffice.ThreadViewState


def jims_thread_list_page() -> rx.Component:
    return jims_backoffice.jims_thread_list_page(header=app_header)
