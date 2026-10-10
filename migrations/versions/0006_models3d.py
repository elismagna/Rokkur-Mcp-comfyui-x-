"""models3d: the 3D studio's models

Revision ID: 0006
Revises: 0005
Create Date: 2026-10-10 18:00:00
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0006'
down_revision = '0005'
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')


def upgrade() -> None:
    op.create_table('models3d',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('method', sa.String(length=32), nullable=False),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('params', _JSON, nullable=False),
    sa.Column('request', _JSON, nullable=False),
    sa.Column('parent_id', sa.String(length=64), nullable=True),
    sa.Column('project_id', sa.String(length=64), nullable=True),
    sa.Column('job_id', sa.String(length=64), nullable=True),
    sa.Column('rel_path', sa.Text(), nullable=True),
    sa.Column('source_rel_path', sa.Text(), nullable=True),
    sa.Column('stats', _JSON, nullable=False),
    sa.Column('render_on', sa.String(length=8), nullable=False),
    sa.Column('remote_id', sa.String(length=64), nullable=True),
    sa.Column('took_s', sa.Float(), nullable=True),
    sa.Column('error', _JSON, nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['parent_id'], ['models3d.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_models3d_parent_id'), 'models3d', ['parent_id'], unique=False)
    op.create_index(op.f('ix_models3d_project_id'), 'models3d', ['project_id'], unique=False)
    op.create_index('ix_models3d_listing', 'models3d', ['status', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_models3d_listing', table_name='models3d')
    op.drop_index(op.f('ix_models3d_project_id'), table_name='models3d')
    op.drop_index(op.f('ix_models3d_parent_id'), table_name='models3d')
    op.drop_table('models3d')
