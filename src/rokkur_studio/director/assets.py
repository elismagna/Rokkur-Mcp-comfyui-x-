"""Asset tracker: the global look and the named characters every shot's prompt is anchored to.

Stored as JSON at ``<data_dir>/director/asset_tracker.json`` using the key names of the studio's
reference format (``GLOBAL_STYLE_MODIFIERS``, ``GLOBAL_NEGATIVE_PROMPT``, ``CHARACTERS``), so it
can be edited by hand, on the dashboard's Director page, or through the API. A character's
description is pasted verbatim into the subject block of every shot, which is what keeps
clothing and appearance from drifting between shots.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_PREFIX = "Cinematic film still"
DEFAULT_STYLE_MODIFIERS = ("35mm anamorphic lens, highly detailed textures, photorealistic "
                           "cinematic film still, depth of field")
DEFAULT_NEGATIVE = "blurry, deformed, drawing, cartoon, illustration, distorted hands"
DEFAULT_CHARACTERS = {
    "NEO": "a 20s athletic male, wearing a tattered black hooded jacket, dark denim jeans, "
           "intense pale features",
}
KEY_RE = re.compile(r"^[A-Z0-9_]{1,40}$")


class TrackerError(ValueError):
    pass


def normalise_key(key: str) -> str:
    """'neo' / 'Old man' -> 'NEO' / 'OLD_MAN'."""
    return re.sub(r"[^A-Z0-9_]", "", re.sub(r"[\s-]+", "_", key.strip().upper()))


class AssetTracker(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    prompt_prefix: str = Field(DEFAULT_PREFIX, alias="PROMPT_PREFIX", max_length=200)
    style_modifiers: str = Field(DEFAULT_STYLE_MODIFIERS, alias="GLOBAL_STYLE_MODIFIERS",
                                 max_length=1000)
    negative_prompt: str = Field(DEFAULT_NEGATIVE, alias="GLOBAL_NEGATIVE_PROMPT",
                                 max_length=1000)
    characters: dict[str, str] = Field(default_factory=lambda: dict(DEFAULT_CHARACTERS),
                                       alias="CHARACTERS")

    @field_validator("characters")
    @classmethod
    def _valid_characters(cls, value: dict[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, desc in value.items():
            k = normalise_key(key)
            if not KEY_RE.match(k):
                raise ValueError(f"character name {key!r} needs letters, digits or _")
            if not desc.strip():
                raise ValueError(f"character {k} has no description")
            if len(desc) > 600:
                raise ValueError(f"character {k}: keep the description under 600 characters")
            out[k] = desc.strip()
        return out

    def character(self, key: str | None) -> str | None:
        return self.characters.get(normalise_key(key)) if key else None

    def to_json(self) -> dict[str, Any]:
        return self.model_dump(by_alias=True)


def tracker_path(data_dir: str | Path) -> Path:
    return Path(data_dir) / "director" / "asset_tracker.json"


def load_tracker(path: Path) -> AssetTracker:
    """The saved tracker, or the defaults when none was saved yet."""
    if not path.is_file():
        return AssetTracker()
    try:
        return AssetTracker.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, ValidationError) as exc:
        raise TrackerError(f"{path} is not a valid asset tracker: {exc}") from exc


def save_tracker(path: Path, tracker: AssetTracker) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(tracker.to_json(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def resolve_character(creative_input: dict[str, Any], tracker: AssetTracker
                      ) -> tuple[str | None, str | None, str | None]:
    """(character key, anchor text, warning) for a project's creative input.

    A tracker character wins; otherwise the project's free-text character description is the
    anchor. An unknown key is reported, not fatal.
    """
    key = creative_input.get("character_key")
    if key:
        text = tracker.character(key)
        if text:
            return normalise_key(key), text, None
        fallback = creative_input.get("character_description") or None
        return None, fallback, (f"character {key!r} is not in the asset tracker; "
                                "no character anchor from the tracker was used")
    return None, creative_input.get("character_description") or None, None
