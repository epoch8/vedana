#!/usr/bin/env python3
"""
Update локального Grist (http://localhost:8484) таблицами из CSV.

Запускается в CI перед тестами. Требует, чтобы контейнер grist уже был запущен и слушал порт 8484.
"""

import os
import time
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import requests


GRIST = os.environ.get("GRIST_URL", "http://localhost:8484")
ORG = os.environ.get("GRIST_ORG", "docs")  # из GRIST_SINGLE_ORG
SEED_ROOT = Path(os.environ.get("GRIST_SEED_DIR", "tests/fixtures/grist"))

DM_DIR = SEED_ROOT / "data_model"
DATA_DIR = SEED_ROOT / "data"

SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})


def wait_ready(timeout=120) -> None:
    """
    Healthcheck для Grist.
    """

    url = f"{GRIST}/api/status"
    deadline = time.time() + timeout

    while time.time() < deadline:
        try:
            r = SESSION.get(url, timeout=2)
            if r.ok:
                return
        except (Exception,) as _:
            pass
        time.sleep(2)

    raise RuntimeError("Grist didn't become ready")


def ensure_workspace(org: str, name: str) -> int:
    """
    Проверка Workspace.
    """

    # org может быть 'current' или домен; в single-org удобно 'current'
    r = SESSION.get(f"{GRIST}/api/orgs/{org}/workspaces")
    r.raise_for_status()
    ws = r.json()
    for w in ws:
        if w["name"] == name:
            return w["id"]
    r = SESSION.post(f"{GRIST}/api/orgs/{org}/workspaces", data=json.dumps({"name": name}))
    r.raise_for_status()
    return int(r.json())


def create_doc(workspace_id: int, name: str) -> str:
    """
    Создаем документ в Grist.
    """

    # создаём пустой документ
    r = SESSION.post(f"{GRIST}/api/workspaces/{workspace_id}/docs", data=json.dumps({"name": name}))
    r.raise_for_status()
    return str(r.json())  # docId (uuid)


def upsert_table(doc_id: str, table_id: str, columns: List[Dict[str, Any]]) -> None:
    """
    Метод для создания таблицы в Grist.
    """

    # создаём таблицу с нужными колонками (если нет)
    r = SESSION.get(f"{GRIST}/api/docs/{doc_id}/tables")
    r.raise_for_status()

    existing = [t["tableId"] for t in r.json()]
    if table_id not in existing:
        payload = [{"tableId": table_id, "columns": columns}]
        rr = SESSION.post(f"{GRIST}/api/docs/{doc_id}/tables", data=json.dumps(payload))
        rr.raise_for_status()


def load_csv_rows(csv_path: Path) -> List[Dict[str, Any]]:
    """
    Метод для чтения CSV и подготовки данных в Grist.
    """

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # пустые -> None, чтобы Grist не ломал типы
            rows.append({k: (v if v != "" else None) for k, v in row.items()})
        return rows


def insert_records(doc_id: str, table_id: str, rows: List[Dict[str, Any]]) -> None:
    """
    Метод для добавления записей в документ Grist.
    """

    if not rows:
        return

    # Grist принимает {"records":[{"fields":{...}}, ...]}
    records = [{"fields": r} for r in rows]

    # Записываем частями по 1000.
    for i in range(0, len(records), 1000):
        chunk = records[i:i+1000]
        rr = SESSION.post(f"{GRIST}/api/docs/{doc_id}/tables/{table_id}/records", data=json.dumps({"records": chunk}))
        rr.raise_for_status()


def update_data_model(doc_id: str) -> None:
    """
    Формируем Data Model.
    """

    # описываем схемы колонок как в тестовой DM
    schema = {
        "Anchors": [
            {"id": "noun", "type": "Text"},
            {"id": "description", "type": "Text"},
            {"id": "id_example", "type": "Text"},
            {"id": "query", "type": "Text"},
        ],
        "Attributes": [
            {"id": "attribute_name", "type": "Text"},
            {"id": "description", "type": "Text"},
            {"id": "anchor", "type": "Text"},
            {"id": "link", "type": "Text"},
            {"id": "data_example", "type": "Text"},
            {"id": "embeddable", "type": "Bool"},
            {"id": "query", "type": "Text"},
            {"id": "dtype", "type": "Text"},
            {"id": "embed_threshold", "type": "Numeric"},
        ],
        "Links": [
            {"id": "anchor1", "type": "Text"},
            {"id": "anchor2", "type": "Text"},
            {"id": "sentence", "type": "Text"},
            {"id": "description", "type": "Text"},
            {"id": "query", "type": "Text"},
            {"id": "anchor1_link_column_name", "type": "Text"},
            {"id": "anchor2_link_column_name", "type": "Text"},
            {"id": "has_direction", "type": "Bool"},
        ],
    }
    for table, cols in schema.items():
        upsert_table(doc_id, table, cols)
        rows = load_csv_rows(DM_DIR / f"{table}.csv")
        insert_records(doc_id, table, rows)


def update_data(doc_id: str) -> None:
    """
    Формируем Data.
    """

    # перечисляем csv и строим схему «всё текстом/числом/булем» — Grist сам приведёт
    for p in DATA_DIR.glob("*.csv"):
        table = p.stem

        with p.open(encoding="utf-8") as f:
            header = next(csv.reader(f))

        # Для всех колонок используем тип Text, Grist SQL вернет как надо (для тестов хватит значени, а не типизации).
        cols = [{"id": h, "type": "Text"} for h in header]

        upsert_table(doc_id, table, cols)
        rows = load_csv_rows(p)
        insert_records(doc_id, table, rows)


def main():
    wait_ready()
    ws_id = ensure_workspace("current", "ci-workspace")
    dm_doc = create_doc(ws_id, "data-model")
    update_data_model(dm_doc)
    data_doc = create_doc(ws_id, "data")
    update_data(data_doc)

    # выведем doc_id'ы для экспорта в окружение pytest
    print(json.dumps({"DM_DOC_ID": dm_doc, "DATA_DOC_ID": data_doc}))


if __name__ == "__main__":
    main()
