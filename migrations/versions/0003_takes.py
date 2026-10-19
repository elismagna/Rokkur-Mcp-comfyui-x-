"""takes: image and video generations from the Create page

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-19 17:00:00
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0003'
down_revision = '0002'
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')


def upgrade() -> None:
    op.create_table('takes',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('mode', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('prompt', sa.Text(), nullable=False),
    sa.Column('final_prompt', sa.Text(), nullable=False),
    sa.Column('negative_prompt', sa.Text(), nullable=False),
    sa.Column('camera', _JSON, nullable=False),
    sa.Column('settings', _JSON, nullable=False),
    sa.Column('workflow', sa.String(length=64), nullable=False),
    sa.Column('render_on', sa.String(length=16), nullable=False),
    sa.Column('start_image', sa.Text(), nullable=True),
    sa.Column('output', sa.Text(), nullable=True),
    sa.Column('thumbnail', sa.Text(), nullable=True),
    sa.Column('parent_id', sa.String(length=64), nullable=True),
    sa.Column('job_id', sa.String(length=64), nullable=True),
    sa.Column('remote_id', sa.String(length=64), nullable=True),
    sa.Column('rating', sa.Integer(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('duration_s', sa.Float(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_takes_status'), 'takes', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_takes_status'), table_name='takes')
    op.drop_table('takes')
