"""Project-level subject lock: one main subject for the whole video.

The vision pass names what is most prominent in each shot's middle frame. A prop in front of
the camera (a bright bucket at the start of a shot) can then be named instead of the person
the video is about, and that shot's prompt switches identity. The lock picks one main subject
for the project from evidence across every shot, and a shot that saw something else keeps the
main subject and lists what it saw with the surroundings.

Evidence, in order: your own words (the scene prompt or a character description) naming
something the shots show, then the person or animal seen in the most shots. With neither,
nothing is locked and each shot keeps what it saw. A character anchor already fixes the
subject, so the lock is not used with one.

This decides the prompt only. The keep mask (``pipeline.subject``) is a salient-object model
that does not read the prompt; locking which object it follows needs a segmentation model
seeded by this subject, which is not built yet.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from rokkur_studio.agents.schemas import ShotPlan

# People and animals: what a viewer follows from shot to shot. Words are folded into families,
# so "fisherman" in one shot and "man in a green shirt" in the next are the same subject, and so
# are "monkey" and "chimp" (a vision model rarely tells those apart).
_FAMILIES = {
    "person": "man men woman women person persons people boy boys girl girls child children kid "
              "kids baby babies guy guys lady ladies gentleman fisherman fishermen fisher sailor "
              "dancer actor actress singer musician player rider driver worker farmer chef cook "
              "teenager toddler grandfather grandmother grandpa grandma father mother dad mom",
    "ape": "monkey monkeys ape apes chimpanzee chimpanzees chimp chimps gorilla gorillas "
           "orangutan orangutans baboon baboons",
    "dog": "dog dogs puppy puppies", "cat": "cat cats kitten kittens",
    "horse": "horse horses pony ponies", "rabbit": "rabbit rabbits bunny bunnies",
    "bird": "bird birds parrot parrots owl owls eagle eagles hawk hawks",
}
LIVING: dict[str, str] = {w: family for family, words in _FAMILIES.items() for w in words.split()}
_ANIMALS = ("cow bull sheep goat pig duck chicken rooster goose swan fish shark dolphin whale seal "
            "otter bear deer fox wolf mouse rat hamster lion tiger leopard cheetah elephant "
            "giraffe zebra camel frog turtle tortoise lizard snake penguin crab octopus")
for _animal in _ANIMALS.split():
    LIVING[_animal] = LIVING[_animal + "s"] = _animal
LIVING.update({"geese": "goose", "mice": "mouse", "wolves": "wolf", "octopuses": "octopus"})
_WORD = re.compile(r"[a-zà-ÿ]+")
_STOP_WORDS = ("the and with from into onto over under near behind front left right side top "
            "bottom small large big little tall short young old dark light bright white black "
            "brown green blue yellow orange purple pink grey gray golden silver wearing holding "
            "sitting standing looking this that these those their there here keep make made "
            "scene shot video frame style face body movement original colorful colourful "
            "warm cold soft hard retro proportions background")
_STOP = set(_STOP_WORDS.split())


def _words(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


_LINK_WORDS = ("in on at with without near next beside behind under over against inside outside "
            "from into onto through along across around while as who that which whose is are "
            "was were has have by of for to")
_LINK = set(_LINK_WORDS.split())


def head(text: str) -> str:
    """The noun phrase an observation is about: up to its first comma, preposition or verb.

    "man in a green shirt holding a tuba" is about a man; "bright yellow bucket and fishing
    lure on the deck" is about a bucket and a lure; "man swimming in water" is not about water.
    """
    first = re.split(r"[,;:.(]", (text or "").lower(), maxsplit=1)[0]
    out: list[str] = []
    for w in _words(first):
        if w in _LINK or (w.endswith("ing") and len(w) > 5 and out):
            break
        out.append(w)
    return " ".join(out)


def living(text: str) -> list[str]:
    """The families of people and animals named in ``text``, in order."""
    out: list[str] = []
    for w in _words(text):
        key = LIVING.get(w)
        if key and key not in out:
            out.append(key)
    return out


def names(text: str, key: str) -> bool:
    """Whether ``text`` names ``key``: a family of living things, or any other noun."""
    words = _words(text)
    return key in living(text) or key in words or f"{key}s" in words or f"{key}es" in words


@dataclass
class SubjectLock:
    key: str | None = None          # "person", "ape", ... or the noun you named, e.g. "bucket"
    description: str = ""           # the fullest observation of it, for shots that missed it
    source: str = "none"            # "your words", "recurring" or "none"
    kept: list[str] = field(default_factory=list)  # shots that saw something else

    @property
    def label(self) -> str:
        return self.description or self.key or ""


def choose(observed: dict[str, str], user_text: str = "") -> SubjectLock:
    """The project's main subject from each shot's observation (shot id -> text)."""
    seen = {sid: text for sid, text in observed.items() if text.strip()}
    if not seen:
        return SubjectLock()
    heads = {sid: head(text) for sid, text in seen.items()}
    key, source = None, "none"
    mentioned = [w for w in _words(user_text) if len(w) > 2 and w not in _STOP]
    in_shots = [w for w in dict.fromkeys(mentioned)
                if any(names(h, LIVING.get(w, w)) for h in heads.values())]
    if in_shots:
        alive = [w for w in in_shots if w in LIVING]
        key, source = LIVING.get((alive or in_shots)[0], (alive or in_shots)[0]), "your words"
    else:
        counts = Counter(k for h in heads.values() for k in living(h))
        if counts:
            first = {k: i for i, h in reversed(list(enumerate(heads.values())))
                     for k in living(h)}
            key = max(counts, key=lambda k: (counts[k], -first[k]))
            source = "recurring"
    if key is None:
        return SubjectLock()
    matching = [seen[sid].strip() for sid, h in heads.items() if names(h, key)]
    description = max(matching, key=len) if matching else key
    return SubjectLock(key=key, description=description, source=source)


def apply(shots: list[ShotPlan], observed: dict[str, str], lock: SubjectLock) -> list[str]:
    """Give each shot the vision pass looked at its subject, keeping the main one. Returns
    warnings for you to review; ``lock.kept`` lists the shots where the main subject was kept.

    An observation that names the main subject anywhere ("yellow bucket next to a man") is
    used as it is. One that saw only something else keeps the main subject when the story puts
    it in the shot, and what was seen goes with the surroundings. When the main subject is not
    in the shot at all, the shot keeps what it saw: writing a person into a prompt over a
    bucket would turn the bucket into a person.
    """
    warnings: list[str] = []
    for shot in shots:
        seen = observed.get(shot.shot_id, "").strip()
        if not seen:
            continue
        if lock.key is None or names(seen, lock.key):
            shot.subject = seen
            continue
        if names(shot.subject, lock.key) or names(shot.intent, lock.key):
            if not names(head(shot.subject), lock.key):
                shot.subject = lock.description
            shot.background = ", ".join(x for x in (seen, shot.background) if x)
            lock.kept.append(shot.shot_id)
            warnings.append(f"{shot.shot_id}: the image read showed “{seen}” first; kept the "
                            f"main subject “{lock.label}” and listed that with the surroundings.")
        else:
            shot.subject = seen
            warnings.append(f"{shot.shot_id}: shows “{seen}” and not the main subject "
                            f"“{lock.label}”; check that this shot is meant to be without it.")
    return warnings
