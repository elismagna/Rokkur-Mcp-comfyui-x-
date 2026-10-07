"""Rights / source eligibility rules. "Found on YouTube" never means "permitted to reuse"."""

from __future__ import annotations

from enum import StrEnum


class RightsCategory(StrEnum):
    USER_OWNED = "USER_OWNED"
    USER_UPLOADED = "USER_UPLOADED"
    CREATOR_PROVIDED = "CREATOR_PROVIDED"
    EXPLICITLY_LICENSED = "EXPLICITLY_LICENSED"
    CREATIVE_COMMONS = "CREATIVE_COMMONS"
    PUBLIC_DOMAIN = "PUBLIC_DOMAIN"
    REFERENCE_ONLY = "REFERENCE_ONLY"
    UNKNOWN = "UNKNOWN"
    REJECTED = "REJECTED"


class RightsStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    NEEDS_HUMAN = "needs_human"
    REJECTED = "rejected"


# Categories whose footage may be ingested and transformed, given evidence.
INGESTIBLE: frozenset[RightsCategory] = frozenset(
    {
        RightsCategory.USER_OWNED,
        RightsCategory.USER_UPLOADED,
        RightsCategory.CREATOR_PROVIDED,
        RightsCategory.EXPLICITLY_LICENSED,
        RightsCategory.CREATIVE_COMMONS,
        RightsCategory.PUBLIC_DOMAIN,
    }
)

# Categories that need a human-recorded permission document to be auto-approved.
NEEDS_EVIDENCE: frozenset[RightsCategory] = frozenset(
    {RightsCategory.CREATOR_PROVIDED, RightsCategory.EXPLICITLY_LICENSED}
)


def evaluate(
    category: RightsCategory,
    *,
    has_evidence: bool,
    block_unknown: bool = True,
    commercial_use: bool | None = None,
) -> tuple[RightsStatus, str]:
    """Deterministic gate. Returns the decision and a human-readable reason."""
    if category is RightsCategory.REJECTED:
        return RightsStatus.REJECTED, "source marked as rejected"
    if category is RightsCategory.REFERENCE_ONLY:
        return RightsStatus.REJECTED, "reference-only sources may be studied but not ingested"
    if category is RightsCategory.UNKNOWN:
        if block_unknown:
            return RightsStatus.NEEDS_HUMAN, "rights unknown: ingestion blocked until a human decides"
        return RightsStatus.NEEDS_HUMAN, "rights unknown"
    if category in NEEDS_EVIDENCE and not has_evidence:
        return RightsStatus.NEEDS_HUMAN, f"{category} requires recorded permission evidence"
    if category is RightsCategory.CREATIVE_COMMONS and commercial_use is False:
        return (
            RightsStatus.NEEDS_HUMAN,
            "Creative Commons licence is non-commercial: confirm the channel is not monetised",
        )
    return RightsStatus.APPROVED, f"{category} is eligible for ingestion and transformation"
