import pytest

from rokkur_studio.domain.rights import RightsCategory as C
from rokkur_studio.domain.rights import RightsStatus as R
from rokkur_studio.domain.rights import evaluate


@pytest.mark.parametrize("category,evidence,expected", [
    (C.USER_OWNED, False, R.APPROVED),
    (C.USER_UPLOADED, False, R.APPROVED),
    (C.PUBLIC_DOMAIN, False, R.APPROVED),
    (C.CREATIVE_COMMONS, False, R.APPROVED),
    (C.CREATOR_PROVIDED, False, R.NEEDS_HUMAN),
    (C.CREATOR_PROVIDED, True, R.APPROVED),
    (C.EXPLICITLY_LICENSED, False, R.NEEDS_HUMAN),
    (C.EXPLICITLY_LICENSED, True, R.APPROVED),
    (C.UNKNOWN, True, R.NEEDS_HUMAN),
    (C.REFERENCE_ONLY, True, R.REJECTED),
    (C.REJECTED, True, R.REJECTED),
])
def test_rights_gate(category, evidence, expected):
    status, reason = evaluate(category, has_evidence=evidence)
    assert status is expected and reason


def test_non_commercial_cc_needs_human():
    status, _ = evaluate(C.CREATIVE_COMMONS, has_evidence=True, commercial_use=False)
    assert status is R.NEEDS_HUMAN
