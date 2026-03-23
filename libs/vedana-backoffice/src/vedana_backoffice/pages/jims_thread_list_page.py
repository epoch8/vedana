from jims_backoffice.pages.jims_thread_list_page import ThreadListState as ThreadListState
from jims_backoffice.pages.jims_thread_list_page import jims_thread_list_page as _base_page

from vedana_backoffice.ui import app_header


def jims_thread_list_page():
    return _base_page(header_fn=app_header)
