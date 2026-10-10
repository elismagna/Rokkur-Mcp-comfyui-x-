"""audio_clips: generated, edited and extracted sound

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-10 12:00:00
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = '0004'
down_revision = '0003'
branch_labels = None
depends_on = None

_JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), 'postgresql')


def upgrade() -> None:
    op.create_table('audio_clips',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('profile', sa.String(length=32), nullable=True),
    sa.Column('workflow', sa.String(length=64), nullable=True),
    sa.Column('prompt', sa.Text(), nullable=False),
    sa.Column('lyrics', sa.Text(), nullable=False),
    sa.Column('params', _JSON, nullable=False),
    sa.Column('request', _JSON, nullable=False),
    sa.Column('parent_id', sa.String(length=64), nullable=True),
    sa.Column('project_id', sa.String(length=64), nullable=True),
    sa.Column('job_id', sa.String(length=64), nullable=True),
    sa.Column('rel_path', sa.Text(), nullable=True),
    sa.Column('duration_s', sa.Float(), nullable=True),
    sa.Column('sample_rate', sa.Integer(), nullable=True),
    sa.Column('channels', sa.Integer(), nullable=True),
    sa.Column('seed', sa.BigInteger(), nullable=True),
    sa.Column('render_on', sa.String(length=8), nullable=False),
    sa.Column('remote_id', sa.String(length=64), nullable=True),
    sa.Column('took_s', sa.Float(), nullable=True),
    sa.Column('error', _JSON, nullable=True),
    sa.Column('verdict', sa.Integer(), nullable=True),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['parent_id'], ['audio_clips.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_audio_clips_parent_id'), 'audio_clips', ['parent_id'], unique=False)
    op.create_index(op.f('ix_audio_clips_project_id'), 'audio_clips', ['project_id'], unique=False)
    op.create_index('ix_audio_clips_listing', 'audio_clips', ['status', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_audio_clips_listing', table_name='audio_clips')
    op.drop_index(op.f('ix_audio_clips_project_id'), table_name='audio_clips')
    op.drop_index(op.f('ix_audio_clips_parent_id'), table_name='audio_clips')
    op.drop_table('audio_clips')
