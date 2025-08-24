import abc
import io
import sqlite3
import time
from ast import literal_eval
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence, Union

import grist_api
import pandas as pd
import requests

from vedana_core.data_model import Attribute, DataModel, Link
from vedana_core.utils import cast_dtype


@dataclass
class Table:
    columns: list[str]
    rows: Sequence[Union[tuple, NamedTuple]]


@dataclass
class AnchorRecord:
    id: str
    type: str
    data: dict[str, Any]
    dp_id: int | None = None


@dataclass
class LinkRecord:
    id_from: str
    id_to: str
    anchor_from: str
    anchor_to: str
    type: str
    data: dict[str, Any]


class DataProvider:
    def get_anchors(self, type_: str, dm_attrs: list[Attribute], dm_anchor_links: list[Link]) -> list[AnchorRecord]:
        raise NotImplementedError("get_anchors must be implemented in subclass")

    def get_links(self, table_name: str, link: Link) -> list[LinkRecord]:
        raise NotImplementedError("get_links must be implemented in subclass")

    def get_anchor_tables(self) -> list[str]:
        raise NotImplementedError("get_anchor_tables must be implemented in subclass")

    def list_anchor_tables(self) -> list[str]:
        raise NotImplementedError("list_anchor_tables must be implemented in subclass")

    def get_link_tables(self) -> list[str]:
        raise NotImplementedError("get_link_tables must be implemented in subclass")

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        self.close()


class CsvDataProvider(DataProvider):
    anchor_file_prefix = "anchor_"
    link_file_prefix = "link_"

    def __init__(self, csv_dir: Path, data_model: DataModel) -> None:
        self.csv_dir = Path(csv_dir)
        self.data_model = data_model
        self._anchor_files: dict[str, Path] = {}
        self._link_files: dict[str, Path] = {}
        self._scan_csv_files()

    def _scan_csv_files(self):
        for file in self.csv_dir.glob("*.csv"):
            fname = file.name.lower()
            if fname.startswith(self.anchor_file_prefix):
                # anchor_type is after prefix and before .csv
                anchor_type = fname[len(self.anchor_file_prefix) : -4]
                self._anchor_files[anchor_type] = file
            elif fname.startswith(self.link_file_prefix):
                link_type = fname[len(self.link_file_prefix) : -4]
                self._link_files[link_type] = file

    def list_anchor_tables(self) -> list[str]:
        return list(self._anchor_files.keys())

    def get_link_tables(self) -> list[str]:
        # todo: link_files != link types
        return list(self._link_files.keys())

    def get_anchors(self, type_: str, dm_attrs: list[Attribute], dm_anchor_links: list[Link]) -> list[AnchorRecord]:
        file = self._anchor_files.get(type_)
        if not file:
            return []
        df = pd.read_csv(file)
        anchors = []
        attrs_dtypes = {a.name: a.dtype for a in self.data_model.attrs}
        grouped = df.groupby(["node_id", "node_type"])
        for (node_id, node_type), group in grouped:
            data = {
                k: cast_dtype(v, k, dtype=attrs_dtypes.get(k))
                for k, v in zip(group["attribute_key"], group["attribute_value"])
            }
            anchors.append(AnchorRecord(str(node_id), str(node_type), data))
        return anchors

    def get_links(self, table_name: str, link: Link) -> list[LinkRecord]:
        file = self._link_files.get(table_name)
        if not file:
            return []
        df = pd.read_csv(file)
        links = []
        grouped = df.groupby(["from_node_id", "to_node_id", "edge_label"])
        for (id_from, id_to, edge_label), group in grouped:
            data = {k: v for k, v in zip(group["attribute_key"], group["attribute_value"]) if not pd.isna(k)}
            links.append(
                LinkRecord(str(id_from), str(id_to), link.anchor_from.noun, link.anchor_to.noun, str(edge_label), data)
            )
        return links

    def close(self) -> None:
        pass


class GristDataProvider(DataProvider):
    anchor_table_prefix = "Anchor_"
    link_table_prefix = "Link_"

    def get_anchor_tables(self) -> list[str]:
        prefix_len = len(self.anchor_table_prefix)
        return [t[prefix_len:] for t in self.list_anchor_tables()]

    @abc.abstractmethod
    def list_anchor_tables(self) -> list[str]: ...

    @abc.abstractmethod
    def list_link_tables(self) -> list[str]: ...

    @abc.abstractmethod
    def get_table(self, table_name: str) -> pd.DataFrame: ...

    def get_anchor_types(self) -> list[str]:
        prefix_len = len(self.anchor_table_prefix)
        return [t[prefix_len:] for t in self.list_anchor_tables()]

    def get_link_tables(self) -> list[str]:
        prefix_len = len(self.link_table_prefix)
        return [t[prefix_len:] for t in self.list_link_tables()]

    def get_anchors(self, type_: str, dm_attrs: list[Attribute], dm_anchor_links: list[Link]) -> list[AnchorRecord]:
        table_name = f"{self.anchor_table_prefix}{type_}"
        table = self.get_table(table_name)
        if "id" not in table.columns:
            table["id"] = [None] * table.shape[0]
        table = table.to_dict(orient="records")

        id_key = f"{type_}_id"
        anchor_ids = set()
        anchors: list[AnchorRecord] = []

        def flatten_list_cells(el):
            if isinstance(el, list) and el[0] == "L":  # flatten List type fields
                if len(el) == 2:
                    return el[1]
                return el[1:]
            elif pd.isna(el):
                return None
            return el

        for row_dict in table:
            db_id = flatten_list_cells(row_dict.pop("id"))
            id_ = row_dict.pop(id_key, None)
            if id_ is None:
                id_ = f"{type_}:{db_id}"
            if pd.isna(id_):
                # print(f"{type_}:{db_id} has {id_key}=nan, skipping")
                continue

            row_dict = {k: flatten_list_cells(v) for k, v in row_dict.items()}
            row_dict = {k: v for k, v in row_dict.items() if not isinstance(v, (bytes, type(None)))}

            if id_ not in anchor_ids:
                anchor_ids.add(id_)
            else:
                print(f"Duplicate anchor id {id_} in table {table_name}\nduplicate data: {row_dict}\n record skipped.")
                continue

            anchors.append(AnchorRecord(id_, type_, row_dict, dp_id=db_id))  # type: ignore

        return anchors

    def get_links(self, table_name: str, link: Link) -> list[LinkRecord]:
        table_name = f"{self.link_table_prefix}{table_name}"
        table = self.get_table(table_name)
        if "id" not in table.columns:
            table["id"] = [None] * table.shape[0]
        table = table.to_dict(orient="records")

        def flatten_list_cells(el):
            if isinstance(el, list) and "L" in el and len(el) == 2:  # flatten List type fields
                el = el[1]
            return el

        links: list[LinkRecord] = []
        for row_dict in table:
            id_from = row_dict.pop("from_node_id")
            id_to = row_dict.pop("to_node_id")
            type_ = row_dict.pop("edge_label")
            row_dict.pop("id")

            if not id_from or not id_to or not type_ or pd.isna(type_) or pd.isna(id_from) or pd.isna(id_to):
                continue
            row_dict = {str(k): flatten_list_cells(v) for k, v in row_dict.items()}
            row_dict = {
                k: v for k, v in row_dict.items() if not isinstance(v, (bytes, list, type(None))) and not pd.isna(v)
            }

            links.append(LinkRecord(id_from, id_to, link.anchor_from.noun, link.anchor_to.noun, type_, row_dict))

        return links


class GristOfflineDataProvider(GristDataProvider):
    """
    Load from grist backup file (.grist)
    """

    def __init__(self, sqlite_path: Path) -> None:
        super().__init__()
        self._conn = sqlite3.connect(sqlite_path)

    def close(self):
        self._conn.close()

    def list_anchor_tables(self) -> list[str]:
        rows = self._conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' and name like '{self.anchor_table_prefix}%'"
        ).fetchall()
        return [row[0] for row in rows]

    def list_link_tables(self) -> list[str]:
        rows = self._conn.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' and name like '{self.link_table_prefix}%'"
        ).fetchall()
        return [row[0] for row in rows]

    def get_table(self, table_name: str) -> pd.DataFrame:
        cur = self._conn.execute(f"SELECT * FROM {table_name}")
        columns = [t[0] for t in cur.description]
        rows = cur.fetchall()
        return pd.DataFrame.from_records(rows, columns=columns)


class GristCsvDataProvider(GristDataProvider):
    def __init__(self, doc_id: str, grist_server: str, api_key: str | None = None) -> None:
        super().__init__()
        self.doc_id = doc_id
        self.grist_server = grist_server
        self.api_key = api_key
        self._table_list: list[str] = self._fetch_table_list()

    def _fetch_table_list(self) -> list[str]:
        url = f"{self.grist_server}/api/docs/{self.doc_id}/tables"
        resp = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"})
        resp.raise_for_status()
        return [t["id"] for t in resp.json()["tables"]]

    def get_table(self, table_id: str) -> pd.DataFrame:
        url = f"{self.grist_server}/api/docs/{self.doc_id}/download/csv?tableId={table_id}"
        resp = requests.get(url, headers={"Authorization": f"Bearer {self.api_key}"}, timeout=600)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        if "id" not in df.columns:
            df = df.reset_index(drop=False, names=["id"])
        return df

    def _list_tables_with_prefix(self, prefix: str) -> list[str]:
        return [t for t in self._table_list if t.startswith(prefix)]

    def list_anchor_tables(self) -> list[str]:
        return self._list_tables_with_prefix(self.anchor_table_prefix)

    def list_link_tables(self) -> list[str]:
        return self._list_tables_with_prefix(self.link_table_prefix)


class GristAPIDataProvider(GristDataProvider):
    def __init__(self, doc_id: str, grist_server: str, api_key: str | None = None) -> None:
        super().__init__()
        self.grist_server = grist_server
        self.doc_id = doc_id
        self.api_key = api_key
        self._client = grist_api.GristDocAPI(doc_id, api_key=api_key, server=grist_server)

    def _list_tables_with_prefix(self, prefix: str) -> list[str]:
        resp = self._client.tables()
        if not resp:
            return []
        table_ids: list[str] = [table["id"] for table in resp.json()["tables"]]
        return [t_id for t_id in table_ids if t_id.startswith(prefix)]

    def list_anchor_tables(self) -> list[str]:
        return self._list_tables_with_prefix(self.anchor_table_prefix)

    def list_link_tables(self) -> list[str]:
        return self._list_tables_with_prefix(self.link_table_prefix)

    def _list_table_columns(self, table_name: str) -> list[str]:
        resp = self._client.columns(table_name)
        if not resp:
            return []
        return [column["id"] for column in resp.json()["columns"]]

    def get_table(self, table_name: str) -> pd.DataFrame:
        # currently will break on tables with non-literal types (links, formulas, etc.)
        columns = self._list_table_columns(table_name)
        rows = self._client.fetch_table(table_name)
        rows = [{c: getattr(r, c) for c in columns} for r in rows]  # filter usage
        return pd.DataFrame(rows, columns=columns)


class GristSQLDataProvider(GristDataProvider):
    """fetches rows via the /sql endpoint in chunks"""

    def __init__(
        self,
        doc_id: str,
        grist_server: str,
        api_key: str | None = None,
        *,
        batch_size: int = 800,
    ) -> None:
        super().__init__()
        self.doc_id = doc_id
        self.grist_server = grist_server
        self.api_key = api_key
        self.batch_size = max(1, batch_size)
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
        self._client = grist_api.GristDocAPI(doc_id, server=grist_server, api_key=api_key)

    def reset_doc(self, doc_id: str):
        print("reopening doc")
        resp = requests.post(
            f"{self.grist_server}/api/docs/{doc_id}/force-reload",
            headers=self.headers,
            timeout=120,
        )
        resp.raise_for_status()

    def clean_doc_history(self, doc_id: str):
        print("cleaning memory")
        resp = requests.post(
            f"{self.grist_server}/api/docs/{doc_id}/states/remove",
            headers=self.headers,
            json={"keep": 1},
            timeout=120,
        )
        resp.raise_for_status()

    def _sql_endpoint(self) -> str:
        """Fully qualified URL of the /sql POST endpoint for this document."""
        return f"{self.grist_server}/api/docs/{self.doc_id}/sql"

    def _run_sql(self, sql: str, att=1) -> list[dict]:
        try:
            resp = requests.post(
                self._sql_endpoint(),
                json={"sql": sql},
                headers=self.headers,
                timeout=180,
            )
            resp.raise_for_status()
            data = resp.json()  # { "records": [ { "fields": {...} }, ... ] }.
            data = data.get("records", [])
        except Exception as e:
            print(f"Failed to fetch {sql}: {e}\n Retrying in 120 seconds...")  # todo logging
            time.sleep(120)
            self.clean_doc_history(self.doc_id)
            self.reset_doc(self.doc_id)
            if att < 3:
                att += 1
                data = self._run_sql(sql, att)
            else:
                raise e
        return data

    def _list_tables_with_prefix(self, prefix: str) -> list[str]:
        resp = self._client.tables()
        if not resp:
            return []
        table_ids: list[str] = [table["id"] for table in resp.json()["tables"]]
        return [t_id for t_id in table_ids if t_id.startswith(prefix)]

    def list_anchor_tables(self) -> list[str]:
        return self._list_tables_with_prefix(self.anchor_table_prefix)

    def list_link_tables(self) -> list[str]:
        return self._list_tables_with_prefix(self.link_table_prefix)

    def _iter_table_rows(self, table_name: str):
        offset = 0
        while True:
            sql = f'SELECT * FROM "{table_name}" ORDER BY id LIMIT {self.batch_size} OFFSET {offset}'
            batch = self._run_sql(sql)
            if not batch:
                break
            for record in batch:
                yield record.get("fields", {})
            offset += self.batch_size

    def get_anchors(self, type_: str, dm_attrs: list[Attribute], dm_anchor_links: list[Link]) -> list[AnchorRecord]:
        dtypes = {e.name: e.dtype for e in dm_attrs}
        table_name = f"{self.anchor_table_prefix}{type_}"
        default_id_key = "node_id"
        fk_links_cols = [c.anchor_from_link_attr_name for c in dm_anchor_links]

        anchors: dict[str, AnchorRecord] = {}

        def flatten(el):
            if isinstance(el, list):
                if el:
                    if el[0] == "L":
                        return el[1] if len(el) == 2 else el[1:]
            elif pd.isna(el):
                return None
            return el

        def safe_fk_list_convert(el):
            try:
                if isinstance(el, str):
                    el = literal_eval(el)
            except SyntaxError:
                print(f"str el not list: {el}")
            return el

        for row in self._iter_table_rows(table_name):
            db_id = flatten(row.get("id"))
            _id = row.get(f"{type_}_id") or row.get(default_id_key) or f"{type_}:{db_id}"
            if pd.isna(_id) or (isinstance(_id, dict) and _id.get("type") == "Buffer"):
                continue

            row.pop("id", None)
            row.pop("node_type", None)
            row.pop(default_id_key, None)
            # row.pop(f"{type_}_id", None)

            # grist meta columns
            row.pop("manualSort", None)

            row = {
                k: v
                for k, v in row.items()
                if not (isinstance(v, dict) and v.get("type") == "Buffer")
                and not k.startswith("gristHelper_")
                and not pd.isna(v)  # type: ignore
            }

            # foreign key link columns - either int id or a list of int ids cast to string
            for col in fk_links_cols:
                if col in row:
                    row[col] = safe_fk_list_convert(row[col])

            clean_data = {
                k: cast_dtype(flatten(v), k, dtypes.get(k))
                for k, v in row.items()
                if not isinstance(v, (bytes, type(None))) and v != ""
            }

            if _id in anchors:  # node already present - populate its attributes
                # anchors[_id].data.update(clean_data)
                print(f"duplicate {type_} id {_id}, skipping...")
            else:
                anchors[_id] = AnchorRecord(str(_id), type_, clean_data, dp_id=db_id)  # type: ignore

        return list(anchors.values())

    def get_links(self, table_name: str, link: Link) -> list[LinkRecord]:
        table_name = f"{self.link_table_prefix}{table_name}"
        columns_resp = self._client.columns(table_name)
        if not columns_resp:
            return []

        def flatten(el):
            if isinstance(el, list) and "L" in el and len(el) == 2:
                el = el[1]
            return el

        links: list[LinkRecord] = []
        for row in self._iter_table_rows(table_name):
            id_from = row.get("id_from") or row.get("from_node_id")
            id_to = row.get("id_to") or row.get("to_node_id")
            edge_label = row.get("type") or row.get("edge_label")
            if not id_from or not id_to or not edge_label or pd.isna(edge_label) or pd.isna(id_from) or pd.isna(id_to):
                continue

            row.pop("manualSort", None)
            row.pop("from_node_id", None)
            row.pop("to_node_id", None)
            row.pop("edge_label", None)
            row.pop("id_from", None)
            row.pop("id_to", None)
            row.pop("type", None)
            row.pop("id", None)

            clean_data = {
                k: flatten(v)
                for k, v in row.items()
                if not isinstance(v, (bytes, list))
                and not pd.isna(v)
                and not (isinstance(v, str) and v == "")
                and not k.startswith("gristHelper_")
            }

            links.append(
                LinkRecord(
                    str(id_from),
                    str(id_to),
                    link.anchor_from.noun,
                    link.anchor_to.noun,
                    str(edge_label),
                    clean_data,
                )
            )

        return links

    def get_table(self, table_name: str) -> pd.DataFrame:
        """
        Table with 1st batch_size rows.
        To iterate over the remaining rows use _iter_table_rows
        """
        cols_resp = self._client.columns(table_name)
        columns: list[str] = []
        if cols_resp:
            columns = [c["id"] for c in cols_resp.json()["columns"]]

        rows_data = []
        for i, row in enumerate(self._iter_table_rows(table_name)):
            if i >= self.batch_size:
                break
            # row dict to tuple in deterministic column order
            rows_data.append(tuple(row.get(col) for col in columns))

        return pd.DataFrame(rows_data, columns=columns)

