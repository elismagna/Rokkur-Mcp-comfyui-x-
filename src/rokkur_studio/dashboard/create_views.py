"""The Create page (/ui/create): Image, Video and Cinema tabs and the takes they made.

Modelled on Open Higgsfield's studios (docs/create.md), running on this studio's own
ComfyUI. Every take goes through services/create.py, the same as the API.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from rokkur_studio.dashboard.views import Ctx, Db, _back, _page
from rokkur_studio.db.models import Take
from rokkur_studio.services import create

router = APIRouter(prefix="/ui/create", include_in_schema=False)
TABS = ("image", "video", "cinema")


def _take(session: Db, take_id: str) -> Take:
    take = session.get(Take, take_id)
    if take is None:
        raise HTTPException(404, "no such take")
    return take


@router.get("", response_class=HTMLResponse)
def create_page(request: Request, ctx: Ctx, session: Db, tab: str = "image",
                start: str = "") -> HTMLResponse:
    start_take = session.get(Take, start) if start else None
    if start_take is not None and (start_take.mode != "image" or not start_take.output):
        start_take = None
    takes = create.listing(session)
    cloud = ctx.settings.cloud
    return _page(request, "create.html", ctx, tab=tab if tab in TABS else "image",
                 takes=takes, start_take=start_take, cinema=create.CINEMA,
                 cinema_labels=create.CINEMA_LABELS, image_sizes=create.IMAGE_SIZES,
                 video_sizes=create.VIDEO_SIZES, seconds=create.SECONDS,
                 max_batch=create.MAX_BATCH, cloud=cloud if cloud.ready else None,
                 preview=ctx.settings.render.renderer != "comfyui",
                 busy=any(t["status"] in ("queued", "running") for t in takes))


@router.get("/takes")
def takes_json(session: Db) -> dict[str, Any]:
    """The take grid's data, polled by the page while something is rendering."""
    return {"takes": create.listing(session)}


@router.post("")
def create_from_form(ctx: Ctx, session: Db, prompt: Annotated[str, Form()],
                     tab: Annotated[str, Form()] = "image",
                     output: Annotated[str, Form()] = "still",
                     aspect: Annotated[str, Form()] = "",
                     seconds: Annotated[int, Form()] = 5,
                     seed: Annotated[str, Form()] = "",
                     count: Annotated[int, Form()] = 1,
                     negative_prompt: Annotated[str, Form()] = "",
                     render_on: Annotated[str, Form()] = "local",
                     start_take: Annotated[str, Form()] = "",
                     start_file: Annotated[UploadFile | None, File()] = None,
                     cam_shot: Annotated[str, Form()] = "",
                     cam_angle: Annotated[str, Form()] = "",
                     cam_move: Annotated[str, Form()] = "",
                     cam_camera: Annotated[str, Form()] = "",
                     cam_lens: Annotated[str, Form()] = "",
                     cam_focal: Annotated[str, Form()] = "",
                     cam_aperture: Annotated[str, Form()] = "",
                     cam_light: Annotated[str, Form()] = "") -> RedirectResponse:
    tab = tab if tab in TABS else "image"
    back = f"/ui/create?tab={tab}"
    clip = tab == "video" or (tab == "cinema" and output == "clip")
    camera = ({"shot": cam_shot, "angle": cam_angle, "move": cam_move, "camera": cam_camera,
               "lens": cam_lens, "focal": cam_focal, "aperture": cam_aperture,
               "light": cam_light} if tab == "cinema" else {})
    try:
        seed_value = int(seed) if seed.strip() else None
    except ValueError:
        return _back(back, err="The seed must be a whole number, or blank for a random one.")
    with tempfile.TemporaryDirectory() as tmp:
        start: Path | None = None
        if clip and start_file is not None and start_file.filename:
            suffix = Path(start_file.filename).suffix.lower()
            if suffix not in create.IMAGE_EXTS:
                return _back(back, err="Use a PNG, JPG or WebP start picture.")
            start = Path(tmp) / f"start{suffix}"
            with start.open("wb") as fh:
                shutil.copyfileobj(start_file.file, fh)
        elif clip and start_take:
            picked = session.get(Take, start_take)
            if picked is None or picked.mode != "image" or not picked.output:
                return _back(back, err="That picture is no longer there.")
            start = create.path_for(ctx.settings, picked.output)
        mode = ("animate" if start is not None else "video") if clip else "image"
        req = create.TakeRequest(mode=mode, prompt=prompt, aspect=aspect, seconds=seconds,
                                 seed=seed_value, count=count, negative_prompt=negative_prompt,
                                 camera=camera, start_image=start, render_on=render_on,
                                 parent_id=start_take or None)
        try:
            takes = create.submit(session, ctx.settings, req)
        except create.CreateError as exc:
            return _back(back, err=str(exc))
    what = "picture" if mode == "image" else "clip"
    n = len(takes)
    return _back(back, msg=f"Queued {n} {what}{'s' if n > 1 else ''}. "
                            "It appears below when it is ready.")


def _file(ctx: Ctx, rel: str | None) -> FileResponse:
    if not rel:
        raise HTTPException(404, "not made yet")
    path = create.path_for(ctx.settings, rel)
    if not path.is_file():
        raise HTTPException(404, "file missing")
    return FileResponse(path)


@router.get("/takes/{take_id}/file")
def take_file(take_id: str, ctx: Ctx, session: Db) -> FileResponse:
    return _file(ctx, _take(session, take_id).output)


@router.get("/takes/{take_id}/thumb")
def take_thumb(take_id: str, ctx: Ctx, session: Db) -> FileResponse:
    return _file(ctx, _take(session, take_id).thumbnail)


@router.get("/takes/{take_id}/start")
def take_start(take_id: str, ctx: Ctx, session: Db) -> FileResponse:
    return _file(ctx, _take(session, take_id).start_image)


@router.post("/takes/{take_id}/again")
def take_again(take_id: str, ctx: Ctx, session: Db) -> RedirectResponse:
    take = _take(session, take_id)
    tab = "cinema" if take.camera else ("image" if take.mode == "image" else "video")
    try:
        create.again(session, ctx.settings, take)
    except create.CreateError as exc:
        return _back(f"/ui/create?tab={tab}", err=str(exc))
    return _back(f"/ui/create?tab={tab}", msg="Queued another take with the next seed.")


@router.post("/takes/{take_id}/rate")
def take_rate(take_id: str, session: Db, value: Annotated[int, Form()]) -> dict[str, Any]:
    take = _take(session, take_id)
    try:
        create.rate(take, value)
    except create.CreateError as exc:
        raise HTTPException(422, str(exc)) from exc
    return {"rating": take.rating}


@router.post("/takes/{take_id}/delete")
def take_delete(take_id: str, ctx: Ctx, session: Db) -> RedirectResponse:
    take = _take(session, take_id)
    tab = "cinema" if take.camera else ("image" if take.mode == "image" else "video")
    create.delete(session, ctx.settings, take)
    return _back(f"/ui/create?tab={tab}", msg="Take deleted.")

