"""images: generated, edited and extracted still pictures

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-10 03:00:00
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
    op.create_table('images',
    sa.Column('id', sa.String(length=64), nullable=False),
    sa.Column('kind', sa.String(length=16), nullable=False),
    sa.Column('status', sa.String(length=16), nullable=False),
    sa.Column('profile', sa.String(length=32), nullable=True),
    sa.Column('workflow', sa.String(length=64), nullable=True),
    sa.Column('prompt', sa.Text(), nullable=False),
    sa.Column('params', _JSON, nullable=False),
    sa.Column('request', _JSON, nullable=False),
    sa.Column('parent_id', sa.String(length=64), nullable=True),
    sa.Column('project_id', sa.String(length=64), nullable=True),
    sa.Column('shot_id', sa.String(length=32), nullable=True),
    sa.Column('job_id', sa.String(length=64), nullable=True),
    sa.Column('rel_path', sa.Text(), nullable=True),
    sa.Column('source_rel_path', sa.Text(), nullable=True),
    sa.Column('width', sa.Integer(), nullable=True),
    sa.Column('height', sa.Integer(), nullable=True),
    sa.Column('seed', sa.BigInteger(), nullable=True),
    sa.Column('render_on', sa.String(length=8), nullable=False),
    sa.Column('remote_id', sa.String(length=64), nullable=True),
    sa.Column('duration_s', sa.Float(), nullable=True),
    sa.Column('error', _JSON, nullable=True),
    sa.Column('verdict', sa.Integer(), nullable=True),
    sa.Column('title', sa.String(length=200), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['parent_id'], ['images.id'], ),
    sa.ForeignKeyConstraint(['project_id'], ['projects.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_images_parent_id'), 'images', ['parent_id'], unique=False)
    op.create_index(op.f('ix_images_project_id'), 'images', ['project_id'], unique=False)
    op.create_index('ix_images_listing', 'images', ['status', 'created_at'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_images_listing', table_name='images')
    op.drop_index(op.f('ix_images_project_id'), table_name='images')
    op.drop_index(op.f('ix_images_parent_id'), table_name='images')
    op.drop_table('images')
