"""The QC stage with the picture checker: contact sheets, stability and the AI review."""

from __future__ import annotations

from typing import Any

from rokkur_studio.services.projects import latest_document
from tests.test_pipeline import create, run, status


class SeeingProvider:
    """A vision-capable provider that always gives the same review."""

    name = "fake_vision"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def supports_images(self) -> bool:
        return True

    def generate(self, role, instructions, payload, schema, images=None):  # noqa: ANN001
        self.calls.append({"role": role, "payload": payload, "images": images})
        return schema.model_validate({
            "description": "A clay-style room with the same figure walking left to right.",
            "prompt_adherence": 8, "style_consistency": 7, "subject_identity": 9, "anatomy": 8,
            "steadiness": 6, "issues": [], "notes": "Fine."})


def _report(ctx, pid: str) -> dict[str, Any]:
    with ctx.db.session() as s:
        doc = latest_document(s, pid, "qc_report")
        assert doc is not None
        return doc.data


def test_qc_writes_contact_sheets_and_stability_without_a_vision_model(ctx, sample_video):
    pid = create(ctx, sample_video)  # rule-based agents cannot see images
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    report = _report(ctx, pid)
    for shot in report["shots"]:
        sheet = ctx.store.path_for(shot["contact_sheet"])
        assert sheet.read_bytes().startswith(b"\x89PNG")
        assert shot["stability"] is not None and shot["flicker"] is not None
        assert "cannot see images" in shot["picture_review_skipped"]
        assert shot.get("prompt_adherence") is None
    assert report["stability"] is not None
    assert "prompt_adherence" in report["not_measured"]
    assert report["picture_reviewed"] == []


def test_qc_copies_the_ai_review_in_as_advisory_scores(ctx, sample_video):
    provider = SeeingProvider()
    ctx.extras["vision_provider"] = provider
    pid = create(ctx, sample_video)
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    report = _report(ctx, pid)
    shots = report["shots"]
    assert len(provider.calls) == len(shots)
    assert all(c["images"] and c["images"][0].startswith(b"\x89PNG") for c in provider.calls)
    assert all(s["prompt_adherence"] == 8 and s["hand_body_deformation"] == 8 for s in shots)
    assert shots[0]["picture_review"]["by"] == "fake_vision"
    assert report["picture_reviewed"] == [s["shot_id"] for s in shots]
    assert "prompt_adherence" not in report["not_measured"]


def test_picture_review_can_be_switched_off_per_project(ctx, sample_video):
    provider = SeeingProvider()
    ctx.extras["vision_provider"] = provider
    pid = create(ctx, sample_video, picture_review=False)
    run(ctx)
    report = _report(ctx, pid)
    assert provider.calls == []
    assert all("picture_review_skipped" not in s and s["contact_sheet"] for s in report["shots"])
