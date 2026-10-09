"""What works for you: a preference profile learned from your ratings, open to inspection.

This is counting, not model training. Every human rating adds the features of what it rated
(prompt terms, workflow, controls, framing). For each feature the studio compares your
ratings where it was present with your average rating:

- Ratings are averaged per project first, so one project rated shot by shot counts once.
- The difference is shrunk toward zero (``SHRINK`` imaginary projects with no effect), so a
  feature seen in one or two projects cannot look decisive.
- A feature is only suggested once it appears in ``MIN_PROJECTS`` rated projects.

Suggestions are offered on the New video form for you to apply. Nothing here changes a
default or a render by itself, and a model's own estimate (``rater="ai"``) is never read.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from statistics import fmean
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from rokkur_studio.db.models import Project, Rating
from rokkur_studio.director.vocabulary import PHRASES
from rokkur_studio.services.ratings import TAGS, VALUES

MIN_PROJECTS = 2
SHRINK = 2.0
SUGGEST_LIFT = 0.4
NOISE_LIFT = 0.1  # smaller than this is a term that sits in nearly everything you rated
STRONG_PROJECTS = 4

KINDS: dict[str, str] = {
    "term": "Prompt term",
    "profile": "Quality profile",
    "workflow": "Workflow",
    "subject": "Main subject",
    "reference": "Appearance reference",
    "control_strength": "Source structure",
    "cfg": "Prompt guidance",
    "steps": "Sampling steps",
    "shot_size": "Framing",
    "camera_angle": "Camera angle",
    "camera_movement": "Camera movement",
    "lighting": "Lighting",
}
_VALUE_LABELS = {("subject", "keep"): "keep the real subject",
                 ("subject", "restyle"): "restyle the subject"}
# Shot-level reference descriptions (renderers.py) and the New video form option they match.
_REFERENCE_MODES = {"subject cutout": "cutout", "none": "none", "source first frame": "source"}
_VOCABULARY = {p.lower() for p in PHRASES.values()}
_WEIGHTED = re.compile(r"^\((.*?)(?::\s*[\d.]+)?\)$")


def prompt_terms(prompt: str | None) -> list[str]:
    """Comma-separated terms of a prompt, without token weights or framing vocabulary.

    Framing and lighting come from the director's fixed vocabulary and are learned as their
    own features; long phrases are scene descriptions, not reusable terms.
    """
    out: list[str] = []
    for raw in (prompt or "").split(","):
        term = raw.strip().lower()
        if m := _WEIGHTED.match(term):
            term = m.group(1)
        term = re.sub(r"\s+", " ", term).strip(" .;:()")
        if 3 <= len(term) <= 48 and len(term.split()) <= 6 and term not in _VOCABULARY:
            out.append(term)
    return list(dict.fromkeys(out))


def _number(value: Any, fmt: str) -> str | None:
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return None


def features(snapshot: dict[str, Any]) -> list[tuple[str, str]]:
    """The (kind, value) pairs a rating says something about."""
    out: list[tuple[str, str]] = []
    if snapshot.get("kind") == "shot":
        out += [("term", t) for t in prompt_terms(snapshot.get("prompt"))]
        for kind in ("workflow", "profile", "reference"):
            if snapshot.get(kind):
                out.append((kind, str(snapshot[kind])))
        out += [(k, str(v)) for k, v in (snapshot.get("framing") or {}).items() if v]
        if (cs := _number(snapshot.get("control_strength"), ".2f")) is not None:
            out.append(("control_strength", cs))
        if (cfg := _number(snapshot.get("cfg"), "g")) is not None:
            out.append(("cfg", cfg))
        if snapshot.get("steps") is not None:
            out.append(("steps", str(snapshot["steps"])))
    else:  # a whole video: what the person chose for it
        out += [("term", t) for t in prompt_terms(snapshot.get("theme"))]
        if snapshot.get("profile"):
            out.append(("profile", str(snapshot["profile"])))
        out += [("workflow", str(w)) for w in snapshot.get("workflows") or []]
    if snapshot.get("subject_mode"):
        out.append(("subject", str(snapshot["subject_mode"])))
    return list(dict.fromkeys(out))


@dataclass
class FeatureStat:
    kind: str
    value: str
    ratings: int
    projects: int
    mean: float
    lift: float
    confidence: str
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return _VALUE_LABELS.get((self.kind, self.value), self.value)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "label": self.label, "kind_label": KINDS.get(self.kind, self.kind)}


def _confidence(projects: int, lift: float) -> str:
    if projects >= STRONG_PROJECTS and abs(lift) >= 0.5:
        return "strong"
    return "some" if projects >= MIN_PROJECTS else "early"


def build_profile(session: Session) -> dict[str, Any]:
    rows = session.execute(select(Rating, Project.name).join(Project, Project.id == Rating.project_id)
                           .where(Rating.rater == "human").order_by(Rating.updated_at)).all()
    by_project: dict[str, list[int]] = defaultdict(list)
    per_feature: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    examples: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    liked_tags: Counter[str] = Counter()
    disliked_tags: Counter[str] = Counter()
    qc_rows: list[tuple[int, str, float | None, dict[str, Any]]] = []
    for rating, name in rows:
        by_project[rating.project_id].append(rating.value)
        (liked_tags if rating.value > 0 else disliked_tags).update(rating.tags or [])
        example = {"project_id": rating.project_id, "project": name, "target": rating.target,
                   "value": rating.value, "render_id": rating.render_id}
        for key in features(rating.snapshot or {}):
            per_feature[key][rating.project_id].append(rating.value)
            if len(examples[key]) < 4:
                examples[key].append(example)
        qc = (rating.snapshot or {}).get("qc") or {}
        if (rating.snapshot or {}).get("kind") == "shot" and qc.get("decision"):
            qc_rows.append((rating.value, qc["decision"], qc.get("overall"), example))
    if not rows:
        return {"count": 0, "projects": 0, "liked": [], "disliked": [], "settings": [],
                "values": {label: 0 for label in VALUES.values()}, "tags": {}, "qc": None}
    average = fmean(fmean(v) for v in by_project.values())
    stats: list[FeatureStat] = []
    for (kind, value), projects in per_feature.items():
        means = [fmean(v) for v in projects.values()]
        lift = sum(m - average for m in means) / (len(means) + SHRINK)
        stats.append(FeatureStat(kind=kind, value=value,
                                 ratings=sum(len(v) for v in projects.values()),
                                 projects=len(means), mean=round(fmean(means), 2),
                                 lift=round(lift, 2), confidence=_confidence(len(means), lift),
                                 examples=examples[(kind, value)]))
    terms = [s for s in stats if s.kind == "term"]
    settings = [s for s in stats if s.kind != "term"]
    counts = Counter(r.value for r, _ in rows)
    return {
        "count": len(rows),
        "projects": len(by_project),
        "average": round(average, 2),
        "values": {label: counts.get(v, 0) for v, label in VALUES.items()},
        "liked": [s.to_dict() for s in sorted((s for s in terms if s.lift >= NOISE_LIFT),
                                              key=lambda s: (-s.lift, -s.projects))[:20]],
        "disliked": [s.to_dict() for s in sorted((s for s in terms if s.lift <= -NOISE_LIFT),
                                                 key=lambda s: (s.lift, -s.projects))[:20]],
        "settings": [s.to_dict() for s in sorted(settings, key=lambda s: (s.kind, -s.lift))],
        "tags": {"liked": {TAGS[t]: n for t, n in liked_tags.most_common() if t in TAGS},
                 "disliked": {TAGS[t]: n for t, n in disliked_tags.most_common() if t in TAGS}},
        "qc": _qc_agreement(qc_rows),
    }


def _qc_agreement(rows: list[tuple[int, str, float | None, dict[str, Any]]]) -> dict[str, Any] | None:
    """How often QC's pass/fail matched your like/dislike, and where it did not."""
    if not rows:
        return None
    agree = sum(1 for value, decision, _, _ in rows if (value > 0) == (decision == "PASS"))
    liked = [o for v, _, o, _ in rows if v > 0 and o is not None]
    disliked = [o for v, _, o, _ in rows if v < 0 and o is not None]
    return {
        "rated": len(rows),
        "agree": agree,
        "percent": round(100 * agree / len(rows)),
        "too_strict": [{**e, "overall": o} for v, d, o, e in rows if v > 0 and d != "PASS"][:8],
        "too_lenient": [{**e, "overall": o} for v, d, o, e in rows if v < 0 and d == "PASS"][:8],
        "liked_mean": round(fmean(liked), 2) if liked else None,
        "disliked_mean": round(fmean(disliked), 2) if disliked else None,
    }


def suggestions(profile: dict[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    """Changes for the New video form, each with the evidence behind it. Applied only by you."""
    out: list[dict[str, Any]] = []

    def why(s: dict[str, Any]) -> str:
        direction = "higher" if s["lift"] > 0 else "lower"
        return (f"rated {abs(s['lift']):.1f} {direction} than your average across "
                f"{s['projects']} projects ({s['ratings']} ratings)")

    def ready(s: dict[str, Any]) -> bool:
        return s["confidence"] != "early" and abs(s["lift"]) >= SUGGEST_LIFT

    for s in profile.get("liked", []):
        if ready(s):
            out.append({"field": "theme", "action": "append", "value": s["value"],
                        "label": f"Add “{s['value']}” to the visual style", "why": why(s),
                        "confidence": s["confidence"]})
    for s in profile.get("disliked", []):
        if ready(s):
            out.append({"field": "negative_prompt", "action": "append", "value": s["value"],
                        "label": f"Avoid “{s['value']}”", "why": why(s),
                        "confidence": s["confidence"]})
    best: dict[str, dict[str, Any]] = {}
    for s in profile.get("settings", []):
        if ready(s) and s["lift"] > 0 and s["lift"] > best.get(s["kind"], {}).get("lift", 0):
            best[s["kind"]] = s
    for kind, s in best.items():
        field_value = _form_setting(kind, s["value"])
        if field_value is not None:
            name, value, text = field_value
            out.append({"field": name, "action": "set", "value": value, "label": text,
                        "why": why(s), "confidence": s["confidence"]})
    order = {"strong": 0, "some": 1}
    out.sort(key=lambda o: order.get(o["confidence"], 2))
    return out[:limit]


def _form_setting(kind: str, value: str) -> tuple[str, str, str] | None:
    """(form field, value, label) for a setting the New video form can take, else None."""
    if kind == "profile":
        return "render_profile", value, f"Render with {value.replace('_', ' ')}"
    if kind == "subject" and value in ("keep", "restyle"):
        return "subject", value, ("Keep the real subject" if value == "keep"
                                  else "Restyle the subject too")
    if kind == "reference" and value in _REFERENCE_MODES:
        return "reference_mode", _REFERENCE_MODES[value], f"Use {value} as the reference"
    if kind == "control_strength":
        return "control_strength", value, f"Follow source structure at {float(value):g}"
    if kind == "cfg":
        return "cfg", value, f"Prompt guidance {value}"
    if kind == "steps" and value.isdigit() and 8 <= int(value) <= 40:
        return "steps", value, f"{value} sampling steps"
    return None


def report(profile: dict[str, Any]) -> str:
    """A plain-text summary to paste to a collaborator: no media, paths or ids."""
    if not profile.get("count"):
        return "No ratings yet."
    lines = [f"Ratings: {profile['count']} across {profile['projects']} projects "
             f"(average {profile['average']:+.2f} on -2..+2)",
             "Distribution: " + ", ".join(f"{k} {v}" for k, v in profile["values"].items())]
    for title, key in (("Liked terms", "liked"), ("Disliked terms", "disliked")):
        items = [s for s in profile[key] if s["confidence"] != "early"][:10]
        if items:
            lines.append(f"{title}: " + "; ".join(
                f"{s['value']} ({s['lift']:+.2f}, {s['projects']} projects)" for s in items))
    settings = [s for s in profile["settings"] if s["confidence"] != "early"]
    if settings:
        lines.append("Settings: " + "; ".join(
            f"{s['kind_label']} {s['label']} ({s['lift']:+.2f}, {s['projects']} projects)"
            for s in settings[:16]))
    for side in ("liked", "disliked"):
        if profile["tags"].get(side):
            lines.append(f"Tags on {side} renders: " + ", ".join(
                f"{t} {n}" for t, n in profile["tags"][side].items()))
    if qc := profile.get("qc"):
        lines.append(f"QC agreed with you on {qc['agree']} of {qc['rated']} rated shots "
                     f"({qc['percent']}%); QC failed {len(qc['too_strict'])} you liked, passed "
                     f"{len(qc['too_lenient'])} you disliked. Mean QC score: liked "
                     f"{qc['liked_mean']}, disliked {qc['disliked_mean']}.")
    return "\n".join(lines)
