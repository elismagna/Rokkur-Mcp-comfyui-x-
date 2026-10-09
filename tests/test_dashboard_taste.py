"""Rating, redoing and the taste page, driven through the dashboard like a person would."""

from sqlalchemy import select

from rokkur_studio.db.models import Rating
from tests.test_dashboard import client_for
from tests.test_pipeline import create, run, status
from tests.test_ratings import _seed_ratings, renders

JSON = {"Accept": "application/json"}


def test_rating_from_the_page_saves_with_or_without_script(ctx, sample_video):
    c = client_for(ctx)
    pid = create(ctx, sample_video)
    run(ctx)
    page = c.get(f"/ui/projects/{pid}").text
    assert "How is it?" in page and "data-rate" in page and "Redo this shot" in page
    r = c.post(f"/ui/projects/{pid}/rate", headers=JSON,
               data={"target": "shot_001", "value": "2", "tags": ["style", "motion"]})
    assert r.status_code == 200 and r.json()["value"] == 2 and r.json()["label"] == "Super like"
    bad = c.post(f"/ui/projects/{pid}/rate", headers=JSON,
                 data={"target": "shot_001", "value": "1", "tags": ["vibes"]})
    assert bad.status_code == 422 and "unknown tags" in bad.json()["error"]
    with ctx.db.session() as s:
        row = s.scalars(select(Rating).where(Rating.project_id == pid)).one()
        assert (row.value, row.tags, row.created_by) == (2, ["style", "motion"], "dashboard")
        assert row.render_id is not None  # the attempt you were looking at
    # Without script the form posts and comes back to the page.
    r = c.post(f"/ui/projects/{pid}/rate", data={"target": "video", "value": "-1"},
               follow_redirects=False)
    assert r.status_code == 303 and "Dislike%20saved" in r.headers["location"]
    assert 'aria-pressed="true"' in c.get(f"/ui/projects/{pid}").text
    for view in ("grid", "list"):
        assert 'class="verdict v-1"' in c.get(f"/ui/projects?view={view}").text
    r = c.post(f"/ui/projects/{pid}/rate", headers=JSON, data={"target": "video", "value": "0"})
    assert r.json() == {"value": 0, "label": "Rating removed", "tags": [], "note": None}


def test_redo_button_renders_only_the_picked_shots_again(ctx, sample_video):
    c = client_for(ctx)
    pid = create(ctx, sample_video)
    run(ctx)
    before = renders(ctx, pid)
    r = c.post(f"/ui/projects/{pid}/redo", follow_redirects=False)
    assert "err=" in r.headers["location"]
    c.post(f"/ui/projects/{pid}/rate", headers=JSON,
           data={"target": "shot_001", "value": "-2", "tags": ["motion"]})
    r = c.post(f"/ui/projects/{pid}/redo", data={"shots": ["shot_001"]}, follow_redirects=False)
    assert "Redoing%20shot%20001" in r.headers["location"]
    run(ctx)
    assert status(ctx, pid) == "READY_TO_PUBLISH"
    after = renders(ctx, pid)
    assert ("shot_001", 2, "succeeded") in after
    assert [x for x in after if x[0] != "shot_001"] == [x for x in before if x[0] != "shot_001"]
    page = c.get(f"/ui/projects/{pid}").text
    assert "What the last redo changed" in page and "shot_001" in page


def test_taste_page_suggestions_and_live_status(ctx, sample_video):
    c = client_for(ctx)
    assert "Nothing learned yet" in c.get("/ui/taste").text
    pid = create(ctx, sample_video)
    run(ctx)
    assert "waiting for your verdict" in c.get("/ui").text
    taste_page = c.get("/ui/taste").text
    assert f'/ui/projects/{pid}/rate' in taste_page and 'name="render_id"' in taste_page
    _seed_ratings(ctx, sample_video, 3, "claymation", 2, cs=0.85)
    _seed_ratings(ctx, sample_video, 1, "claymation", 2, cs=0.85)
    _seed_ratings(ctx, sample_video, 2, "neon noir", -2)
    taste_page = c.get("/ui/taste").text
    assert "claymation" in taste_page and "Try next" in taste_page and "taste-report" in taste_page
    new = c.get("/ui/new").text
    assert 'data-apply data-field="theme" data-action="append" data-value="claymation"' in new
    live = c.get("/ui/status").json()
    assert live == {"running": [], "queued": 0, "approvals": live["approvals"]}
    css = c.get("/ui/static/studio.css")
    assert css.status_code == 200 and css.headers["content-type"].startswith("text/css")
    assert "/ui/static/studio.js?v=" in c.get("/ui").text
