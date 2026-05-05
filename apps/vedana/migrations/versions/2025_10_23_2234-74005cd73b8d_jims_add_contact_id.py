"""jims_add_contact_id

Revision ID: 74005cd73b8d
Revises: 7e6b7d0b54d1
Create Date: 2025-10-23 22:34:18.265911

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "74005cd73b8d"
down_revision: Union[str, None] = "7e6b7d0b54d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("threads", sa.Column("contact_id", sa.String(), nullable=True))
    op.create_index(op.f("ix_threads_contact_id"), "threads", ["contact_id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("threads", "contact_id")
