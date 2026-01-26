from __future__ import annotations

import json
from typing import Any

import reflex as rx

from config_plane.impl.sql import create_sql_config_repo
from vedana_core.settings import settings as core_settings


def _payload_to_tables(payload: dict[str, Any]) -> list[dict[str, Any]]:
    anchors = payload.get("anchors", []) or []
    links = payload.get("links", []) or []
    queries = payload.get("queries", []) or []
    prompts = payload.get("prompts", []) or []
    lifecycle = payload.get("conversation_lifecycle", []) or []

    anchor_rows = [
        {
            "noun": a.get("noun", ""),
            "description": a.get("description", ""),
            "id_example": a.get("id_example", ""),
            "query": a.get("query", ""),
            "attributes": len(a.get("attributes", []) or []),
        }
        for a in anchors
    ]

    anchor_attr_rows: list[dict[str, Any]] = []
    for a in anchors:
        for attr in a.get("attributes", []) or []:
            anchor_attr_rows.append(
                {
                    "anchor": a.get("noun", ""),
                    "attribute_name": attr.get("attribute_name", ""),
                    "description": attr.get("description", ""),
                    "data_example": attr.get("data_example", ""),
                    "embeddable": bool(attr.get("embeddable", False)),
                    "query": attr.get("query", ""),
                    "dtype": attr.get("dtype", ""),
                    "embed_threshold": attr.get("embed_threshold", ""),
                }
            )

    link_rows = [
        {
            "sentence": l.get("sentence", ""),
            "anchor1": l.get("anchor1", ""),
            "anchor2": l.get("anchor2", ""),
            "description": l.get("description", ""),
            "query": l.get("query", ""),
            "anchor1_link_column_name": l.get("anchor1_link_column_name", ""),
            "anchor2_link_column_name": l.get("anchor2_link_column_name", ""),
            "has_direction": bool(l.get("has_direction", False)),
            "attributes": len(l.get("attributes", []) or []),
        }
        for l in links
    ]

    link_attr_rows: list[dict[str, Any]] = []
    for l in links:
        for attr in l.get("attributes", []) or []:
            link_attr_rows.append(
                {
                    "link": l.get("sentence", ""),
                    "attribute_name": attr.get("attribute_name", ""),
                    "description": attr.get("description", ""),
                    "data_example": attr.get("data_example", ""),
                    "embeddable": bool(attr.get("embeddable", False)),
                    "query": attr.get("query", ""),
                    "dtype": attr.get("dtype", ""),
                    "embed_threshold": attr.get("embed_threshold", ""),
                }
            )

    query_rows = [
        {"name": q.get("name", ""), "example": q.get("example", "")} for q in queries
    ]
    prompt_rows = [{"name": p.get("name", ""), "text": p.get("text", "")} for p in prompts]
    lifecycle_rows = [{"event": e.get("event", ""), "text": e.get("text", "")} for e in lifecycle]

    return [
        {
            "name": "Anchors",
            "columns": ["noun", "description", "id_example", "query", "attributes"],
            "rows": anchor_rows,
            "row_count": len(anchor_rows),
        },
        {
            "name": "Anchor attributes",
            "columns": [
                "anchor",
                "attribute_name",
                "description",
                "data_example",
                "embeddable",
                "query",
                "dtype",
                "embed_threshold",
            ],
            "rows": anchor_attr_rows,
            "row_count": len(anchor_attr_rows),
        },
        {
            "name": "Links",
            "columns": [
                "sentence",
                "anchor1",
                "anchor2",
                "description",
                "query",
                "anchor1_link_column_name",
                "anchor2_link_column_name",
                "has_direction",
                "attributes",
            ],
            "rows": link_rows,
            "row_count": len(link_rows),
        },
        {
            "name": "Link attributes",
            "columns": [
                "link",
                "attribute_name",
                "description",
                "data_example",
                "embeddable",
                "query",
                "dtype",
                "embed_threshold",
            ],
            "rows": link_attr_rows,
            "row_count": len(link_attr_rows),
        },
        {
            "name": "Queries",
            "columns": ["name", "example"],
            "rows": query_rows,
            "row_count": len(query_rows),
        },
        {
            "name": "Prompts",
            "columns": ["name", "text"],
            "rows": prompt_rows,
            "row_count": len(prompt_rows),
        },
        {
            "name": "Conversation lifecycle",
            "columns": ["event", "text"],
            "rows": lifecycle_rows,
            "row_count": len(lifecycle_rows),
        },
    ]


def _diff_rows(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
    key_fields: list[str],
) -> list[dict[str, Any]]:
    before_map: dict[tuple, dict[str, Any]] = {
        tuple(row.get(k, "") for k in key_fields): row for row in before
    }
    after_map: dict[tuple, dict[str, Any]] = {
        tuple(row.get(k, "") for k in key_fields): row for row in after
    }
    keys = sorted(set(before_map) | set(after_map))
    rows: list[dict[str, Any]] = []
    for key in keys:
        before_row = before_map.get(key)
        after_row = after_map.get(key)
        if before_row and not after_row:
            status = "removed"
        elif after_row and not before_row:
            status = "added"
        elif before_row == after_row:
            status = "unchanged"
        else:
            status = "changed"
        if status == "unchanged":
            continue
        rows.append(
            {
                "status": status,
                "key": " / ".join(str(k) for k in key),
                "before": json.dumps(before_row or {}, ensure_ascii=False),
                "after": json.dumps(after_row or {}, ensure_ascii=False),
            }
        )
    return rows


def _diff_tables(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_tables = {t["name"]: t for t in _payload_to_tables(left)}
    right_tables = {t["name"]: t for t in _payload_to_tables(right)}
    anchors_rows = _diff_rows(left_tables["Anchors"]["rows"], right_tables["Anchors"]["rows"], ["noun"])
    anchor_attr_rows = _diff_rows(
        left_tables["Anchor attributes"]["rows"],
        right_tables["Anchor attributes"]["rows"],
        ["anchor", "attribute_name"],
    )
    links_rows = _diff_rows(left_tables["Links"]["rows"], right_tables["Links"]["rows"], ["sentence"])
    link_attr_rows = _diff_rows(
        left_tables["Link attributes"]["rows"],
        right_tables["Link attributes"]["rows"],
        ["link", "attribute_name"],
    )
    queries_rows = _diff_rows(left_tables["Queries"]["rows"], right_tables["Queries"]["rows"], ["name"])
    prompts_rows = _diff_rows(left_tables["Prompts"]["rows"], right_tables["Prompts"]["rows"], ["name"])
    lifecycle_rows = _diff_rows(
        left_tables["Conversation lifecycle"]["rows"],
        right_tables["Conversation lifecycle"]["rows"],
        ["event"],
    )

    return [
        {
            "name": "Anchors",
            "columns": ["status", "key", "before", "after"],
            "rows": anchors_rows,
            "row_count": len(anchors_rows),
        },
        {
            "name": "Anchor attributes",
            "columns": ["status", "key", "before", "after"],
            "rows": anchor_attr_rows,
            "row_count": len(anchor_attr_rows),
        },
        {
            "name": "Links",
            "columns": ["status", "key", "before", "after"],
            "rows": links_rows,
            "row_count": len(links_rows),
        },
        {
            "name": "Link attributes",
            "columns": ["status", "key", "before", "after"],
            "rows": link_attr_rows,
            "row_count": len(link_attr_rows),
        },
        {
            "name": "Queries",
            "columns": ["status", "key", "before", "after"],
            "rows": queries_rows,
            "row_count": len(queries_rows),
        },
        {
            "name": "Prompts",
            "columns": ["status", "key", "before", "after"],
            "rows": prompts_rows,
            "row_count": len(prompts_rows),
        },
        {
            "name": "Conversation lifecycle",
            "columns": ["status", "key", "before", "after"],
            "rows": lifecycle_rows,
            "row_count": len(lifecycle_rows),
        },
    ]


class DataModelState(rx.State):
    dev_branch: str = core_settings.config_plane_dev_branch
    prod_branch: str = core_settings.config_plane_prod_branch
    dev_snapshot_id: int | None = None
    prod_snapshot_id: int | None = None
    is_loading: bool = False
    error_message: str = ""

    diff_branch_left: str = core_settings.config_plane_dev_branch
    diff_branch_right: str = core_settings.config_plane_prod_branch
    diff_tables: list[dict[str, Any]] = []
    diff_error_message: str = ""
    diff_is_loading: bool = False

    view_branch: str = core_settings.config_plane_dev_branch
    view_snapshot_input: str = ""
    view_tables: list[dict[str, Any]] = []
    view_error_message: str = ""
    view_is_loading: bool = False
    quick_view_snapshot_id: str = ""
    quick_view_tables: list[dict[str, Any]] = []
    quick_view_error_message: str = ""
    quick_view_is_loading: bool = False
    quick_view_open: bool = False

    quick_diff_snapshot_id: str = ""
    quick_diff_tables: list[dict[str, Any]] = []
    quick_diff_error_message: str = ""
    quick_diff_is_loading: bool = False
    quick_diff_open: bool = False

    def set_diff_branch_left(self, value: str) -> None:
        self.diff_branch_left = value

    def set_diff_branch_right(self, value: str) -> None:
        self.diff_branch_right = value

    def set_view_branch(self, value: str) -> None:
        self.view_branch = value

    def set_view_snapshot_input(self, value: str) -> None:
        self.view_snapshot_input = value

    def set_quick_view_open(self, value: bool) -> None:
        self.quick_view_open = bool(value)

    def set_quick_diff_open(self, value: bool) -> None:
        self.quick_diff_open = bool(value)

    @rx.event(background=True)  # type: ignore[operator]
    async def load_status(self):
        async with self:
            self.is_loading = True
            self.error_message = ""
            try:
                svc = ConfigPlaneService(branch=self.dev_branch)
                self.dev_snapshot_id = svc.get_branch_snapshot_id_for(self.dev_branch)
                self.prod_snapshot_id = svc.get_branch_snapshot_id_for(self.prod_branch)
            except Exception as exc:
                self.error_message = str(exc)
                self.dev_snapshot_id = None
                self.prod_snapshot_id = None
            finally:
                self.is_loading = False
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def load_diff(self):
        async with self:
            self.diff_is_loading = True
            self.diff_error_message = ""
            try:
                svc = ConfigPlaneService(branch=self.dev_branch)
                left_id = svc.get_branch_snapshot_id_for(self.diff_branch_left)
                right_id = svc.get_branch_snapshot_id_for(self.diff_branch_right)
                left_blob = svc.get_snapshot_blob(left_id, "vedana.data_model")
                right_blob = svc.get_snapshot_blob(right_id, "vedana.data_model")
                if left_blob is None or right_blob is None:
                    raise RuntimeError("Missing config-plane blob for diff")
                left_payload = json.loads(left_blob.decode("utf-8"))
                right_payload = json.loads(right_blob.decode("utf-8"))
                self.diff_tables = _diff_tables(left_payload, right_payload)
            except Exception as exc:
                self.diff_error_message = str(exc)
                self.diff_tables = []
            finally:
                self.diff_is_loading = False
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def load_view_snapshot(self):
        async with self:
            self.view_is_loading = True
            self.view_error_message = ""
            try:
                svc = ConfigPlaneService(branch=self.view_branch)
                snapshot_id = None
                if self.view_snapshot_input.strip():
                    snapshot_id = int(self.view_snapshot_input.strip())
                else:
                    snapshot_id = svc.get_branch_snapshot_id_for(self.view_branch)
                if snapshot_id is None:
                    raise RuntimeError(f"No snapshot for branch '{self.view_branch}'")
                blob = svc.get_snapshot_blob(snapshot_id, "vedana.data_model")
                if blob is None:
                    raise RuntimeError(f"Snapshot {snapshot_id} missing key 'vedana.data_model'")
                payload = json.loads(blob.decode("utf-8"))
                self.view_tables = _payload_to_tables(payload)
            except Exception as exc:
                self.view_error_message = str(exc)
                self.view_tables = []
            finally:
                self.view_is_loading = False
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def open_quick_view(self, snapshot_id: str | int):
        async with self:
            self.quick_view_open = True
            self.quick_view_is_loading = True
            self.quick_view_error_message = ""
            self.quick_view_snapshot_id = str(snapshot_id or "")
            try:
                snap = int(str(snapshot_id))
                svc = ConfigPlaneService(branch=self.dev_branch)
                blob = svc.get_snapshot_blob(snap, "vedana.data_model")
                if blob is None:
                    raise RuntimeError(f"Snapshot {snap} missing key 'vedana.data_model'")
                payload = json.loads(blob.decode("utf-8"))
                self.quick_view_tables = _payload_to_tables(payload)
            except Exception as exc:
                self.quick_view_error_message = str(exc)
                self.quick_view_tables = []
            finally:
                self.quick_view_is_loading = False
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def open_quick_diff(self, snapshot_id: str | int, compare_branch: str | None = None):
        async with self:
            self.quick_diff_open = True
            self.quick_diff_is_loading = True
            self.quick_diff_error_message = ""
            self.quick_diff_snapshot_id = str(snapshot_id or "")
            try:
                snap = int(str(snapshot_id))
                branch = compare_branch or self.prod_branch
                svc = ConfigPlaneService(branch=branch)
                compare_id = svc.get_branch_snapshot_id_for(branch)
                if compare_id is None:
                    raise RuntimeError(f"No snapshot for branch '{branch}'")
                left_blob = svc.get_snapshot_blob(snap, "vedana.data_model")
                right_blob = svc.get_snapshot_blob(compare_id, "vedana.data_model")
                if left_blob is None or right_blob is None:
                    raise RuntimeError("Missing config-plane blob for diff")
                left_payload = json.loads(left_blob.decode("utf-8"))
                right_payload = json.loads(right_blob.decode("utf-8"))
                self.quick_diff_tables = _diff_tables(left_payload, right_payload)
            except Exception as exc:
                self.quick_diff_error_message = str(exc)
                self.quick_diff_tables = []
            finally:
                self.quick_diff_is_loading = False
            yield

    @rx.event(background=True)  # type: ignore[operator]
    async def sync_prod_with_dev(self):
        async with self:
            try:
                svc = ConfigPlaneService(branch=self.prod_branch)
                snapshot_id = svc.sync_branch(self.prod_branch, self.dev_branch)
                yield rx.toast.success(f"Prod synced to Dev (snapshot {snapshot_id})")
            except Exception as exc:
                self.error_message = str(exc)
                yield rx.toast.error(f"Failed to sync prod: {exc}")
            finally:
                yield DataModelState.load_status()
