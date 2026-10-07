"""FastAPI application factory. OpenAPI is served at /docs and /openapi.json."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from rokkur_studio import __version__
from rokkur_studio.api import routes_director, routes_projects, routes_system, routes_youtube
from rokkur_studio.config import Settings, get_settings
from rokkur_studio.dashboard.views import router as dashboard_router
from rokkur_studio.pipeline.context import StudioContext, build_context


def create_app(settings: Settings | None = None, ctx: StudioContext | None = None) -> FastAPI:
    ctx = ctx or build_context(settings or get_settings())
    app = FastAPI(title="Rökkur Studio", version=__version__,
                  description="Local-first multi-agent AI video production studio")
    app.state.ctx = ctx
    app.include_router(routes_projects.router)
    app.include_router(routes_system.router)
    app.include_router(routes_director.router)
    app.include_router(routes_youtube.router)
    app.include_router(dashboard_router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    return app
