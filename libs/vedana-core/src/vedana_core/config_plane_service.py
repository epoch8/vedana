from __future__ import annotations

from sqlalchemy.orm import sessionmaker

from config_plane.impl.sql import SqlConfigRepo, create_sql_config_repo
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
        return SqlConfigRepo.get_branch_head_for(self.session_maker, branch)

    def get_snapshot_blob(self, snapshot_id: int, key: str) -> bytes | None:
        return SqlConfigRepo.get_snapshot_blob_for(self.session_maker, snapshot_id, key)

    def snapshot_exists(self, snapshot_id: int) -> bool:
        return SqlConfigRepo.snapshot_exists_for(self.session_maker, snapshot_id)

    def get_latest_committed_payload(self, branch: str) -> bytes | None:
        snapshot_id = self.get_branch_snapshot_id_for(branch)
        if snapshot_id is None:
            return None
        return self.get_snapshot_blob(snapshot_id, "vedana.data_model")

    def commit_payload(self, payload: bytes) -> int | None:
        repo = create_sql_config_repo(self.session_maker, branch=self.branch)
        return repo.commit_if_changed("vedana.data_model", payload)

    def sync_branch(self, target_branch: str, source_branch: str) -> int:
        snapshot_id = self.get_branch_snapshot_id_for(source_branch)
        if snapshot_id is None:
            raise ValueError(f"Source branch '{source_branch}' has no commits")
        self.set_branch_snapshot_id(target_branch, snapshot_id)
        return snapshot_id

    def set_branch_snapshot_id(self, branch: str, snapshot_id: int) -> None:
        SqlConfigRepo.set_branch_head_for(self.session_maker, branch, snapshot_id)
