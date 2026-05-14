"""add pgvector and filter indexes

Revision ID: 8b1f2d4c9a71
Revises: 17f7b4956ca5
Create Date: 2026-04-24 12:00:00.000000

"""

import os
from typing import Sequence, Union

from alembic import op

# If True, this migration will create/drop the pgvector extension.
# Set to False (default) when the extension is managed externally (e.g., by cloud provider).
CREATE_PGVECTOR_EXTENSION = os.environ.get("CREATE_PGVECTOR_EXTENSION", "false").lower() == "true"


# revision identifiers, used by Alembic.
revision: str = "8b1f2d4c9a71"
down_revision: Union[str, None] = "17f7b4956ca5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # parallel workers don't load extensions dynamically loaded in the parent session, 
    # which fails the migration on managed services
    if not CREATE_PGVECTOR_EXTENSION:
        op.execute("SET max_parallel_maintenance_workers = 0;")

    """Upgrade schema."""
    # https://github.com/pgvector/pgvector#index-options
    # m - the max number of connections per layer (16 by default)
    # ef_construction - the size of the dynamic candidate list for constructing the graph (64 by default)
    # A higher value of ef_construction provides better recall at the cost of index build time / insert speed.
    # template to provide options (defaults here):
    # ... USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS rag_anchor_embeddings_hnsw_idx
        ON rag_anchor_embeddings
        USING hnsw (embedding vector_cosine_ops)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS rag_edge_embeddings_hnsw_idx
        ON rag_edge_embeddings
        USING hnsw (embedding vector_cosine_ops)
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.execute("DROP INDEX IF EXISTS rag_edge_embeddings_hnsw_idx")
    op.execute("DROP INDEX IF EXISTS rag_anchor_embeddings_hnsw_idx")
