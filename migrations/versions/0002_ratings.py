"""ratings: human (and later model) judgements of renders and videos

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-09 14:30:00
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')


def upgrade() -> None:
    op.create_table('ratings',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('project_id', sa.String(length=64), nullable=False),
    sa.Column('target', sa.String(length=32), nullable=False),
    sa.Column('render_id', sa.String(length=64), nullable=True),
    sa.Column('asset_id', sa.String(length=64), nullable=True),
    sa.Column('rater', sa.String(length=16), nullable=False),
    sa.Column('value', sa.Integer(), nullable=False),
    sa.Column('tags', _JSON, nullable=False),
    sa.Column('note', sa.Text(), nullable=True),
    sa.Column('snapshot', _JSON, nullable=False),
    sa.Column('created_by', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    sa.ForeignKeyConstraint(['asset_id'], ['assets.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.ForeignKeyConstraint(['render_id'], ['renders.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ratings_project_id'), 'ratings', ['project_id'], unique=False)
    op.create_index('ix_ratings_target', 'ratings', ['project_id', 'target', 'rater'],
                    unique=False)


def downgrade() -> None:
    op.drop_index('ix_ratings_target', table_name='ratings')
    op.drop_index(op.f('ix_ratings_project_id'), table_name='ratings')
    op.drop_table('ratings')
