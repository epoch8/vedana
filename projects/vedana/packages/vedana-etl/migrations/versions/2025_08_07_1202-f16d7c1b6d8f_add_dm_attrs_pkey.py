"""add_dm_attrs_pkey

Revision ID: f16d7c1b6d8f
Revises: 0752f3cfee39
Create Date: 2025-08-07 12:02:43.571090

"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "f16d7c1b6d8f"
down_revision = "0752f3cfee39"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint("dm_attributes_pkey", "dm_attributes", type_="primary")
    op.drop_constraint("dm_attributes_meta_pkey", "dm_attributes_meta", type_="primary")

    op.alter_column("dm_attributes", "anchor", existing_type=sa.VARCHAR(), nullable=False)
    op.add_column("dm_attributes_meta", sa.Column("anchor", sa.String(), nullable=False))

    op.create_primary_key("dm_attributes_pkey", "dm_attributes", ["anchor", "attribute_name"])
    op.create_primary_key("dm_attributes_meta_pkey", "dm_attributes_meta", ["anchor", "attribute_name"])


def downgrade() -> None:
    op.drop_constraint("dm_attributes_pkey", "dm_attributes", type_="primary")
    op.drop_constraint("dm_attributes_meta_pkey", "dm_attributes_meta", type_="primary")

    op.drop_column("dm_attributes_meta", "anchor")
    op.alter_column("dm_attributes", "anchor", existing_type=sa.VARCHAR(), nullable=True)

    op.create_primary_key("dm_attributes_pkey", "dm_attributes", ["attribute_name"])
    op.create_primary_key("dm_attributes_meta_pkey", "dm_attributes_meta", ["attribute_name"])
