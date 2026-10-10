"""rea_runs: runs of the REA reverse-engineering CLI

Revision ID: 0005
Revises: 0004
Create Date: 2026-10-10 16:00:00
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0005'
down_revision = '0004'
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')


def upgrade() -> None:
    op.create_table('rea_runs',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('preset', sa.String(length=32), nullable=False),
    sa.Column('target', sa.Text(), nullable=False),
    sa.Column('query', sa.Text(), nullable=False),
    sa.Column('args', _JSON, nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('exit_code', sa.Integer(), nullable=True),
    sa.Column('job_id', sa.String(length=64), nullable=True),
    sa.Column('project_id', sa.String(length=64), nullable=True),
    sa.Column('rel_path', sa.Text(), nullable=True),
    sa.Column('summary', _JSON, nullable=False),
    sa.Column('error', _JSON, nullable=True),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('took_s', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_rea_runs_project_id'), 'rea_runs', ['project_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_rea_runs_project_id'), table_name='rea_runs')
    op.drop_table('rea_runs')
