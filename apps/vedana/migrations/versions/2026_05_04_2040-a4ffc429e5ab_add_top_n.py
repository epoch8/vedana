"""add_top_n

Revision ID: a4ffc429e5ab
Revises: 8b1f2d4c9a71
Create Date: 2026-05-04 20:40:36.858884

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4ffc429e5ab'
down_revision: Union[str, None] = '8b1f2d4c9a71'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('dm_anchor_attributes', sa.Column('embed_top_n', sa.Integer(), nullable=True))
    op.add_column('dm_link_attributes', sa.Column('embed_top_n', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('dm_link_attributes', 'embed_top_n')
    op.drop_column('dm_anchor_attributes', 'embed_top_n')
    # ### end Alembic commands ###
