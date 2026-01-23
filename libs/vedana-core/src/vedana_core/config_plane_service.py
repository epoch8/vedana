from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker

from config_plane.impl.sql import (
    BlobModel,
    BranchModel,
    SnapshotModel,
    SnapshotItemModel,
    create_sql_config_repo,
)
from vedana_core.db import get_db_engine


class ConfigPlaneService:
    """
    Extension of config-plane with some methods that are missing in the package.
    Maybe some of them should be implemented there?
    """

    def __init__(self, branch: str = "master") -> None:
        self.branch = branch
        self.session_maker = sessionmaker(bind=get_db_engine(), expire_on_commit=False)

    def get_branch_snapshot_id(self) -> int | None:
        return self.get_branch_snapshot_id_for(self.branch)

    def get_branch_snapshot_id_for(self, branch: str) -> int | None:
        with self.session_maker() as session:
            branch = session.execute(
                sa.select(BranchModel).where(BranchModel.name == branch)
            ).scalar_one_or_none()
            return branch.snapshot_id

    def get_snapshot_blob(self, snapshot_id: int, key: str) -> bytes | None:
        with self.session_maker() as session:
            item = session.execute(
                sa.select(SnapshotItemModel).where(
                    SnapshotItemModel.snapshot_id == snapshot_id,
                    SnapshotItemModel.key == key,
                )
            ).scalar_one_or_none()
            if item is None or item.blob_id is None:
                return None

            if item.blob:
                return item.blob.content

            blob = session.execute(
                sa.select(BlobModel).where(BlobModel.id == item.blob_id)
            ).scalar_one_or_none()
            return blob.content if blob else None

    def snapshot_exists(self, snapshot_id: int) -> bool:
        with self.session_maker() as session:
            found = session.execute(
                sa.select(SnapshotModel.id).where(SnapshotModel.id == snapshot_id)
            ).scalar_one_or_none()
            return found is not None

    def get_latest_committed_payload(self, branch: str) -> bytes | None:
        snapshot_id = self.get_branch_snapshot_id_for(branch)
        if snapshot_id is None:
            return None
        return self.get_snapshot_blob(snapshot_id, "vedana.data_model")

    def commit_payload(self, payload: bytes) -> int | None:
        latest = self.get_latest_committed_payload(self.branch)  # check changes, skip if unchanged
        if latest == payload:
            return self.get_branch_snapshot_id()
        repo = create_sql_config_repo(self.session_maker, branch=self.branch)
        repo.set("vedana.data_model", payload)
        repo.commit()
        return repo.parent_snapshot.snapshot_id if repo.parent_snapshot else repo.stage_snapshot_id

    def sync_branch(self, target_branch: str, source_branch: str) -> int:
        snapshot_id = self.get_branch_snapshot_id_for(source_branch)
        if snapshot_id is None:
            raise ValueError(f"Source branch '{source_branch}' has no commits")
        self.set_branch_snapshot_id(target_branch, snapshot_id)
        return snapshot_id

    def set_branch_snapshot_id(self, branch: str, snapshot_id: int) -> None:
        with self.session_maker() as session:
            branch_model = session.execute(
                sa.select(BranchModel).where(BranchModel.name == branch)
            ).scalar_one_or_none()
            if branch_model:
                branch_model.snapshot_id = snapshot_id
            else:
                branch_model = BranchModel(name=branch, snapshot_id=snapshot_id)
                session.add(branch_model)
            session.commit()
