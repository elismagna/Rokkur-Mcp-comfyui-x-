"""One main subject per video: a prop seen first must not take over a shot (director.subject)."""

from __future__ import annotations

from rokkur_studio.agents.providers import AgentUnavailable, RuleBasedProvider
from rokkur_studio.agents.roles import CreativeDirector, DirectorOfPhotography
from rokkur_studio.agents.schemas import ShotFraming
from rokkur_studio.config import DirectorSection
from rokkur_studio.director.assets import AssetTracker
from rokkur_studio.director.passes import direct
from rokkur_studio.director.subject import choose, head, living

ANALYSIS = {"duration": 6.0, "fps": 24.0, "width": 360, "height": 640, "source_asset": "p/s.mp4",
            "shots": [{"shot_id": f"shot_00{i}", "start": 2 * (i - 1), "end": 2 * i,
                       "motion_intensity": 0.2, "motion_type": "gentle"} for i in (1, 2, 3)]}

# The case from the field: a bright bucket fills shot 001's middle frame, the man in the boat
# is the subject of the later shots.
BOAT = {"shot_001": "bright yellow bucket and fishing lure on the deck",
        "shot_002": "man in a green shirt and brown hat holding a tuba",
        "shot_003": "fisherman in a brown hat, smiling"}


class Seeing:
    """A vision model that reports one fixed observation per shot."""

    name = "test_vision"

    def __init__(self, seen: dict[str, str]) -> None:
        self.seen, self.n = seen, 0

    def generate(self, role, instructions, payload, schema, images=None):
        if role != DirectorOfPhotography.role:
            raise AgentUnavailable("this fake only reads shots")  # the story falls back to rules
        sid = payload["shot"]["shot_id"]
        return ShotFraming(shot_size="Medium Shot", camera_angle="Eye-level",
                           camera_movement="Static", lighting="High-key overhead",
                           observed_subject=self.seen[sid], observed_background="lake")

    def supports_images(self) -> bool:
        return True


def plan(**creative):
    brief, _ = CreativeDirector(RuleBasedProvider()).run({"theme": "clay", **creative},
                                                         ANALYSIS, "youtube_short")
    return brief


def frames():
    return {s["shot_id"]: b"jpeg" for s in ANALYSIS["shots"]}


def test_what_an_observation_is_about():
    assert head("man in a green shirt holding a tuba") == "man"
    assert head("man swimming in water") == "man"
    assert head("bright yellow bucket and fishing lure on the deck") == "bright yellow bucket and"
    assert living("fisherman and two dogs") == ["person", "dog"]
    assert living("small brown chimp") == living("monkey") == ["ape"]


def test_the_recurring_person_is_the_main_subject_and_the_bucket_stays_a_prop():
    lock = choose(BOAT)
    assert (lock.key, lock.source) == ("person", "recurring")
    assert lock.description == "man in a green shirt and brown hat holding a tuba"
    brief = plan()
    brief.shot_plan[0].intent = "wide view of the man in the boat as a tuba rises from the water"
    dp = DirectorOfPhotography(Seeing(BOAT), vision=True)
    out, by = dp.run(brief, frames())
    assert by == "test_vision+vision"
    first = out.shot_plan[0]
    assert first.subject == "man in a green shirt and brown hat holding a tuba"
    assert first.background.startswith("bright yellow bucket") and "lake" in first.background
    assert [s.subject for s in out.shot_plan[1:]] == [BOAT["shot_002"], BOAT["shot_003"]]
    assert dp.lock.kept == ["shot_001"] and "shot_001" in dp.warnings[0]


def test_a_shot_without_the_main_subject_keeps_what_it_shows():
    # Writing a man into a shot that only shows a bucket would turn the bucket into a man.
    out, _ = (dp := DirectorOfPhotography(Seeing(BOAT), vision=True)).run(plan(), frames())
    assert out.shot_plan[0].subject == BOAT["shot_001"]
    assert dp.lock.kept == [] and "not the main subject" in dp.warnings[0]


def test_your_words_can_make_the_prop_the_subject():
    seen = {**BOAT, "shot_002": "man holding a yellow bucket", "shot_003": "yellow bucket on a dock"}
    dp = DirectorOfPhotography(Seeing(seen), vision=True)
    out, _ = dp.run(plan(), frames(), user_text="The yellow bucket is the hero of this film")
    assert (dp.lock.key, dp.lock.source) == ("bucket", "your words")
    assert out.shot_plan[0].subject == BOAT["shot_001"]
    assert out.shot_plan[1].subject == "man holding a yellow bucket"  # names the bucket: kept
    assert out.shot_plan[2].subject == "yellow bucket on a dock"


def test_no_person_or_animal_means_no_lock():
    seen = {"shot_001": "red car on a road", "shot_002": "empty street", "shot_003": "neon sign"}
    dp = DirectorOfPhotography(Seeing(seen), vision=True)
    out, _ = dp.run(plan(), frames())
    assert dp.lock.key is None and dp.warnings == []
    assert [s.subject for s in out.shot_plan] == list(seen.values())


def test_the_brief_shows_the_main_subject():
    brief, _ = direct(Seeing(BOAT), {"theme": "clay", "prompt": "the man in the boat"},  # type: ignore[arg-type]
                      ANALYSIS, "youtube_short", tracker=AssetTracker(),
                      settings=DirectorSection(), keyframes=frames())
    notes = brief.director
    assert notes is not None and notes.subject == BOAT["shot_002"]
    assert notes.subject_source == "your words"
    assert "man in a green shirt" in brief.shot_plan[1].prompt
    assert any(w.startswith("shot_001: shows") for w in notes.warnings)
