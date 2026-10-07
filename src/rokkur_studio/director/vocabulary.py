"""Industry-standard cinematography library: the allowed choices matrix.

The Director of Photography pass may only pick these exact terms. They are ``Literal`` types, so
the JSON schema sent to Ollama constrains the model's output to them (llama.cpp grammar) and
pydantic rejects anything else. Each term also has the phrase the prompt compiler writes into
the diffusion prompt.
"""

from __future__ import annotations

import re
from typing import Literal, get_args

ShotSize = Literal["Extreme Close-up", "Close-up", "Medium Shot", "Cowboy Shot", "Full Shot",
                   "Extreme Long Shot"]
CameraAngle = Literal["Low-angle", "Eye-level", "High-angle", "Dutch tilt", "Overhead crane",
                      "Birds-eye view"]
CameraMovement = Literal["Static", "Slow push-in", "Tracking pan", "Handheld shake", "Jib tilt",
                         "Dolly zoom"]
Lighting = Literal["Volumetric god rays", "High-key overhead", "Chiaroscuro high-contrast",
                   "Moody neon rim lighting", "Golden hour diffusion", "Cyberpunk bi-color hue",
                   "Rembrandt lighting"]

SHOT_SIZES: tuple[str, ...] = get_args(ShotSize)
CAMERA_ANGLES: tuple[str, ...] = get_args(CameraAngle)
CAMERA_MOVEMENTS: tuple[str, ...] = get_args(CameraMovement)
LIGHTING_STYLES: tuple[str, ...] = get_args(Lighting)

VOCABULARY: dict[str, tuple[str, ...]] = {
    "Shot sizes": SHOT_SIZES,
    "Camera angles": CAMERA_ANGLES,
    "Camera movement": CAMERA_MOVEMENTS,
    "Lighting styles": LIGHTING_STYLES,
}

# How each term reads inside a prompt. Framing terms name the shot ("close-up shot") so the
# text encoder sees a framing noun, which is what the Rule of Nouns puts first.
PHRASES: dict[str, str] = {
    "Extreme Close-up": "extreme close-up shot",
    "Close-up": "close-up shot",
    "Medium Shot": "medium shot",
    "Cowboy Shot": "cowboy shot",
    "Full Shot": "full shot",
    "Extreme Long Shot": "extreme long shot",
    "Low-angle": "low-angle shot",
    "Eye-level": "eye-level shot",
    "High-angle": "high-angle shot",
    "Dutch tilt": "dutch tilt angle",
    "Overhead crane": "overhead crane shot",
    "Birds-eye view": "birds-eye view",
    "Static": "static camera",
    "Slow push-in": "slow push-in",
    "Tracking pan": "tracking pan",
    "Handheld shake": "handheld camera shake",
    "Jib tilt": "jib tilt",
    "Dolly zoom": "dolly zoom",
    "Volumetric god rays": "volumetric god rays",
    "High-key overhead": "high-key overhead lighting",
    "Chiaroscuro high-contrast": "chiaroscuro high-contrast lighting",
    "Moody neon rim lighting": "moody neon rim lighting",
    "Golden hour diffusion": "golden hour diffusion",
    "Cyberpunk bi-color hue": "cyberpunk bi-color hue",
    "Rembrandt lighting": "rembrandt lighting",
}

# -- rule-based picks (used when no model answers) ------------------------------------------
# The source video's measured motion is the only framing fact the analysis has, so the rules
# map it to a movement term and otherwise choose the neutral framing.
MOVEMENT_FOR_MOTION: dict[str, str] = {
    "static": "Static",
    "gentle": "Slow push-in",
    "moderate": "Tracking pan",
    "high": "Handheld shake",
}
DEFAULT_SHOT_SIZE = "Medium Shot"
DEFAULT_ANGLE = "Eye-level"
DEFAULT_LIGHTING = "Golden hour diffusion"

# First match wins, so the more specific looks come first.
LIGHTING_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("cyberpunk", "synthwave", "blade runner", "hologram"), "Cyberpunk bi-color hue"),
    (("neon", "night city", "nightclub", "vaporwave", "arcade"), "Moody neon rim lighting"),
    (("noir", "horror", "gothic", "thriller", "dark fantasy", "vampire"),
     "Chiaroscuro high-contrast"),
    (("baroque", "renaissance", "oil painting", "rembrandt", "portrait"), "Rembrandt lighting"),
    (("forest", "cathedral", "church", "temple", "fog", "mist", "dust", "underwater", "ruins"),
     "Volumetric god rays"),
    (("sunset", "sunrise", "golden", "summer", "autumn", "warm", "desert", "beach"),
     "Golden hour diffusion"),
    (("commercial", "product", "sitcom", "pastel", "bright", "clean", "toy", "kids"),
     "High-key overhead"),
]


def lighting_for(text: str) -> str:
    """Rule-based lighting pick from the theme/style words."""
    low = text.lower()
    for words, lighting in LIGHTING_KEYWORDS:
        if any(re.search(rf"\b{re.escape(w)}\b", low) for w in words):
            return lighting
    return DEFAULT_LIGHTING
