from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
from grist_api import GristDocAPI
from sqlalchemy import Column

from datapipe.run_config import RunConfig
from datapipe.store.database import MetaKey
from datapipe.store.table_store import TableStore, TableStoreCaps
from datapipe.types import DataDF, DataSchema, IndexDF, MetaSchema


@dataclass(frozen=True)
class _GristAuth:
    server: str
    api_key: str
    doc_id: str


def _row_dicts_from_df(df: DataDF, columns: Sequence[str]) -> List[Dict[str, Any]]:
    # Ensure JSON-serializable values: convert NaN to None
    clean_df = df.copy()
    if not clean_df.empty:
        clean_df = clean_df.where(pd.notna(clean_df), None)
    return [
        {col: row.get(col) for col in columns}
        for row in clean_df.to_dict(orient="records")  # type: ignore
    ]


def _pk_tuple_from_fields(fields: Dict[str, Any], pk_columns: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(fields.get(col) for col in pk_columns)


class GristStore(TableStore):
    """Grist document table backend.
    Provided table is expected to exist and its schema is expected to match the provided schema.
    Based on Grist Python Client: https://github.com/gristlabs/py_grist_api
    """

    caps = TableStoreCaps(
        supports_delete=True,
        supports_get_schema=True,
        supports_read_all_rows=True,
        supports_read_nonexistent_rows=True,
        supports_read_meta_pseudo_df=True,
    )

    def __init__(
        self,
        server: str,
        api_key: str,
        doc_id: str,
        table: str,
        data_sql_schema: List[Column],
        page_size: int = 1000,
    ) -> None:
        super().__init__()
        self._auth = _GristAuth(server=server.rstrip("/"), api_key=api_key, doc_id=doc_id)
        self.table = table
        self.data_sql_schema = data_sql_schema
        self._pk_columns: List[str] = [c.name for c in self.data_sql_schema if c.primary_key]
        self._value_columns: List[str] = [c.name for c in self.data_sql_schema if not c.primary_key]
        self._page_size = page_size

        self._api = GristDocAPI(self._auth.doc_id, api_key=self._auth.api_key, server=self._auth.server)

    # Basic metadata interface
    def get_schema(self) -> DataSchema:
        return self.data_sql_schema

    def get_primary_schema(self) -> DataSchema:
        return [c for c in self.data_sql_schema if c.primary_key]

    def get_meta_schema(self) -> MetaSchema:
        meta_key_prop = MetaKey.get_property_name()
        return [column for column in self.data_sql_schema if hasattr(column, meta_key_prop)]

    # Endpoints (for low-level access via client.call)
    @property
    def _records_path(self) -> str:
        return f"tables/{self.table}/records"

    # Low-level helpers
    def _iter_all_records(self) -> Iterator[Dict[str, Any]]:
        offset = 0
        while True:
            records = (
                self._api.call(self._records_path, {"limit": self._page_size, "offset": offset}, method="GET")
                .json()
                .get("records")
            )
            if not records:
                break
            for rec in records:
                yield rec
            if len(records) < self._page_size:
                break
            offset += self._page_size

    def _map_pk_to_record_id(self) -> Dict[Tuple[Any, ...], int]:
        mapping: Dict[Tuple[Any, ...], int] = {}
        for rec in self._iter_all_records():
            rec_id_any = rec.get("id")
            if not isinstance(rec_id_any, int):
                continue
            rec_id: int = rec_id_any
            fields: Dict[str, Any] = rec.get("fields", {})
            pk = _pk_tuple_from_fields(fields, self._pk_columns)
            mapping[pk] = rec_id
        return mapping

    def _fetch_all_rows(self) -> List[Dict[str, Any]]:
        data = self._api.fetch_table(self.table)
        return [r._asdict() for r in data]  # namedtuples -> dict

    # CRUD
    def insert_rows(self, df: DataDF) -> None:
        if df.empty:
            return
        all_columns = self._pk_columns + self._value_columns
        rows = _row_dicts_from_df(df, all_columns)
        self._api.add_records(self.table, rows)

    def update_rows(self, df: DataDF) -> None:
        if df.empty:
            return

        # Upsert using sync_table helper
        all_columns = self._pk_columns + self._value_columns
        new_data = [type("Row", (), row) for row in _row_dicts_from_df(df, all_columns)]
        key_cols = [(c, c) for c in self._pk_columns]
        other_cols = [(c, c) for c in self._value_columns]
        self._api.sync_table(self.table, new_data, key_cols, other_cols)

    def delete_rows(self, idx: IndexDF) -> None:
        if idx is None or idx.empty:
            return

        # Map keys to record ids, then delete via SQL
        pk_to_id = self._map_pk_to_record_id()

        ids_to_delete: List[int] = []
        for row in idx[self._pk_columns].to_dict(orient="records"):
            pk = tuple(row.get(col) for col in self._pk_columns)
            rec_id = pk_to_id.get(pk)
            if rec_id is not None:
                ids_to_delete.append(rec_id)

        if not ids_to_delete:
            return

        self._api.delete_records(self.table, ids_to_delete)

    def read_rows(self, idx: Optional[IndexDF] = None) -> DataDF:
        if idx is None:  # Read full table
            rows = self._fetch_all_rows()
            if rows:
                return pd.DataFrame.from_records(rows)[[c.name for c in self.data_sql_schema]]
            return pd.DataFrame(columns=[c.name for c in self.data_sql_schema])

        if idx.empty:
            return pd.DataFrame(columns=[c.name for c in self.data_sql_schema])

        # Build a local index of existing rows and filter
        wanted: set[Tuple[Any, ...]] = set()
        for row in idx[self._pk_columns].to_dict(orient="records"):
            wanted.add(tuple(row.get(col) for col in self._pk_columns))

        rows = self._fetch_all_rows()
        if not rows:
            return pd.DataFrame(columns=[c.name for c in self.data_sql_schema])
        df_all = pd.DataFrame.from_records(rows)
        if self._pk_columns:
            tuples = list(map(tuple, df_all[self._pk_columns].astype(object).itertuples(index=False, name=None)))
            mask = [t in wanted for t in tuples]
            df_sel = df_all.loc[mask]
        else:
            df_sel = df_all.iloc[0:0]
        if df_sel.empty:
            return pd.DataFrame(columns=[c.name for c in self.data_sql_schema])
        return df_sel[[c.name for c in self.data_sql_schema]]

    def read_rows_meta_pseudo_df(
        self,
        chunksize: int = 1000,
        run_config: Optional[RunConfig] = None,
    ) -> Iterator[DataDF]:
        # Stream records in pages and yield as DataFrames
        buffer: List[Dict[str, Any]] = []

        def flush() -> Iterator[DataDF]:
            nonlocal buffer
            if buffer:
                df = pd.DataFrame.from_records(buffer)
                buffer = []
                yield df

        rows = self._fetch_all_rows()
        for row in rows:
            buffer.append(row)
            if len(buffer) >= chunksize:
                yield from flush()

        yield from flush()
