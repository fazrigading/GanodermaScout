"""Add stable chunk IDs and generated full-text search vectors.

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-08
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("document_chunks", sa.Column("chunk_id", sa.String(512), nullable=True))
    op.execute(
        "UPDATE document_chunks "
        "SET chunk_id = document_id || '#' || id::text "
        "WHERE chunk_id IS NULL"
    )
    op.alter_column(
        "document_chunks",
        "chunk_id",
        existing_type=sa.String(512),
        nullable=False,
    )
    op.create_unique_constraint("uq_document_chunks_chunk_id", "document_chunks", ["chunk_id"])
    op.add_column(
        "document_chunks",
        sa.Column(
            "search_vector",
            postgresql.TSVECTOR(),
            sa.Computed("to_tsvector('english', content)", persisted=True),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_document_chunks_search_vector_gin",
        "document_chunks",
        ["search_vector"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_document_chunks_search_vector_gin", table_name="document_chunks")
    op.drop_column("document_chunks", "search_vector")
    op.drop_constraint("uq_document_chunks_chunk_id", "document_chunks", type_="unique")
    op.drop_column("document_chunks", "chunk_id")
