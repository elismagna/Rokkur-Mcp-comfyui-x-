"""A reviewed prompt draft for the New video form."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from rokkur_studio.agents.providers import AgentProvider


class PromptEditRequest(BaseModel):
    theme: str = Field(min_length=1, max_length=2000)
    prompt: str = Field(default="", max_length=2000)
    subject: Literal["auto", "keep", "restyle"] = "auto"
    character: str = Field(default="", max_length=700)
    target_format: Literal["youtube_short", "youtube_video"] = "youtube_short"


class PromptDraft(BaseModel):
    theme: str = Field(min_length=1, max_length=2000)
    prompt: str = Field(default="", max_length=2000)
    changes: str = Field(default="", max_length=500)


class PromptEditor:
    """Polish visual direction without silently changing the user's submitted form."""

    instructions = (
        "Help the user prepare two short prompts for Wan VACE video restyling. Return a "
        "visual theme and a scene prompt, plus one brief note about the edits. Preserve the "
        "user's intent, named characters, actions, setting, and requested mood. Do not invent "
        "source-video details, new characters, dialogue, text, logos, or camera moves. Keep "
        "style and medium in theme; keep visible subject, pose, action, and setting in prompt. "
        "Respect the requested subject mode and character description. Use concrete visual "
        "language, avoid conflicting adjectives and story narration, and keep each prompt "
        "concise. If a field is blank, keep it blank unless needed to preserve the user's "
        "meaning. The user will review and edit both fields before using them."
    )

    def __init__(self, provider: AgentProvider) -> None:
        self.provider = provider

    def polish(self, request: PromptEditRequest) -> PromptDraft:
        return self.provider.generate(
            "Prompt Editor",
            self.instructions,
            {"theme": request.theme, "prompt": request.prompt, "subject": request.subject,
             "character": request.character, "target_format": request.target_format},
            PromptDraft,
        )
