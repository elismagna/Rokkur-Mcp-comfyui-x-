"""ComfyUI diffusion parsing rules, applied deterministically after the agents answer.

1. Rule of Nouns: framing and nouns first; the subject is one unbroken block (only commas, no
   full stops, colons or brackets inside it). ``clean_phrase`` enforces the punctuation part;
   the prompt compiler enforces the order.
2. Action strip-out: diffusion models read states, not stories. ``strip_actions`` rewrites
   progressive actions ("is walking", "suddenly runs", "thinking about home") into physical
   states ("in mid-stride", "in full sprint", "focused expression").
3. Token weights: only the compiler adds ``(term:1.3)`` weights, to framing terms. Weights or
   brackets inside agent or user text are removed so the prompt's emphasis stays predictable.
"""

from __future__ import annotations

import re

# -- 3. token weights ---------------------------------------------------------------------


def weighted(phrase: str, weight: float) -> str:
    """ComfyUI emphasis syntax; weight 1.0 means no brackets."""
    if abs(weight - 1.0) < 1e-9:
        return phrase
    return f"({phrase}:{weight:g})"


_WEIGHT_IN_BRACKETS = re.compile(r"\(([^()]*?):\s*-?\d+(?:\.\d+)?\s*\)")
_BRACKETS = re.compile(r"[()\[\]{}<>|\\`\"“”]")
_SENTENCE_END = re.compile(r"[.!?;](?=\s|$)|[!?;]|:(?=\s|$)")


def clean_phrase(text: str | None) -> str:
    """Free text made safe for a prompt: no weights or brackets, no sentence punctuation.

    Sentence breaks become commas, so "Hero stands. Rain falls" reads as one comma-separated
    block. Decimals and ratios such as "f/1.8" or "16:9" are left alone.
    """
    if not text:
        return ""
    t = text.replace("’", "'").replace("‘", "'").replace("--neg", " ")
    t = re.sub(r"-{2,}", " ", t)
    t = re.sub(r"\bembedding:", "", t)
    t = _WEIGHT_IN_BRACKETS.sub(r"\1", t)
    t = _BRACKETS.sub(" ", t)
    t = _SENTENCE_END.sub(",", t)
    t = re.sub(r"[\r\n\t]+", ", ", t)
    return tidy(t)


def tidy(text: str) -> str:
    """Collapse whitespace and empty comma segments."""
    parts = [re.sub(r"\s+", " ", p).strip() for p in text.split(",")]
    return ", ".join(p for p in parts if p)


def dedupe_terms(text: str) -> str:
    """Drop repeated comma-separated terms (case-insensitive, ignoring weights), keep order."""
    seen: set[str] = set()
    out = []
    for term in (t.strip() for t in text.split(",")):
        key = _WEIGHT_IN_BRACKETS.sub(r"\1", term).strip("() ").lower()
        if term and key not in seen:
            seen.add(key)
            out.append(term)
    return ", ".join(out)


# -- 2. action strip-out ------------------------------------------------------------------
# verb: (forms: base, 3rd person, -ing, past…), physical state, preposition rewrites.
_PHYSICAL: dict[str, tuple[tuple[str, ...], str, dict[str, str]]] = {
    "walk": (("walk", "walks", "walking", "walked"), "in mid-stride", {}),
    "run": (("run", "runs", "running", "ran"), "in full sprint", {}),
    "sprint": (("sprint", "sprints", "sprinting", "sprinted"), "in full sprint", {}),
    "jog": (("jog", "jogs", "jogging", "jogged"), "mid-jog", {}),
    "jump": (("jump", "jumps", "jumping", "jumped"), "airborne mid-jump", {}),
    "leap": (("leap", "leaps", "leaping", "leapt", "leaped"), "airborne mid-leap", {}),
    "fall": (("fall", "falls", "falling", "fell"), "mid-fall", {}),
    "dance": (("dance", "dances", "dancing", "danced"), "frozen in a dance pose", {}),
    "fight": (("fight", "fights", "fighting", "fought"), "in a fighting stance", {}),
    "punch": (("punch", "punches", "punching", "punched"), "fist extended mid-punch", {}),
    "kick": (("kick", "kicks", "kicking", "kicked"), "leg extended mid-kick", {}),
    "turn": (("turn", "turns", "turning", "turned"), "head turned sharply", {}),
    "look": (("look", "looks", "looking", "looked"), "gaze fixed",
             {"at": "on", "up": "upward", "down": "downward", "back": "backward"}),
    "stare": (("stare", "stares", "staring", "stared"), "gaze fixed", {"at": "on"}),
    "glance": (("glance", "glances", "glancing", "glanced"), "eyes cut sideways",
               {"at": "toward"}),
    "smile": (("smile", "smiles", "smiling", "smiled"), "smiling expression", {"at": "toward"}),
    "laugh": (("laugh", "laughs", "laughing", "laughed"), "laughing expression", {}),
    "cry": (("cry", "cries", "crying", "cried"), "tear-streaked face", {}),
    "scream": (("scream", "screams", "screaming", "screamed"), "mouth wide open mid-scream", {}),
    "shout": (("shout", "shouts", "shouting", "shouted"), "mouth open mid-shout", {}),
    "talk": (("talk", "talks", "talking", "talked"), "mouth open mid-sentence", {}),
    "speak": (("speak", "speaks", "speaking", "spoke"), "mouth open mid-sentence", {}),
    "sing": (("sing", "sings", "singing", "sang"), "mouth open mid-song", {}),
    "reach": (("reach", "reaches", "reaching", "reached"), "arm outstretched", {"for": "toward"}),
    "wave": (("wave", "waves", "waving", "waved"), "hand raised mid-wave", {}),
    "point": (("point", "points", "pointing", "pointed"), "arm extended", {"at": "toward"}),
    "climb": (("climb", "climbs", "climbing", "climbed"), "gripping a ledge mid-climb", {}),
    "swim": (("swim", "swims", "swimming", "swam"), "mid-stroke in water", {}),
    "fly": (("fly", "flies", "flying", "flew"), "airborne", {}),
    "drive": (("drive", "drives", "driving", "drove"), "hands on the steering wheel", {}),
    "ride": (("ride", "rides", "riding", "rode"), "seated astride", {}),
    "sit": (("sit", "sits", "sitting", "sat"), "seated", {}),
    "sleep": (("sleep", "sleeps", "sleeping", "slept"), "eyes closed asleep", {}),
    "eat": (("eat", "eats", "eating", "ate"), "food raised to the mouth", {}),
    "drink": (("drink", "drinks", "drinking", "drank"), "glass raised to the lips", {}),
    "throw": (("throw", "throws", "throwing", "threw"), "arm extended mid-throw", {}),
    "spin": (("spin", "spins", "spinning", "spun"), "mid-spin", {}),
    "crawl": (("crawl", "crawls", "crawling", "crawled"), "on hands and knees", {}),
    "hide": (("hide", "hides", "hiding", "hid"), "crouched in cover", {}),
    "stop": (("stop", "stops", "stopping", "stopped"), "frozen mid-step", {}),
}
# Inner states become an expression; whatever they were about is dropped (a model cannot
# render "thinking about his childhood").
_MENTAL: dict[str, tuple[tuple[str, ...], str]] = {
    "think": (("think", "thinks", "thinking", "thought"), "focused expression"),
    "ponder": (("ponder", "ponders", "pondering", "pondered"), "focused expression"),
    "consider": (("consider", "considers", "considering", "considered"), "focused expression"),
    "contemplate": (("contemplate", "contemplates", "contemplating", "contemplated"),
                    "contemplative expression"),
    "wonder": (("wonder", "wonders", "wondering", "wondered"), "contemplative expression"),
    "remember": (("remember", "remembers", "remembering", "remembered"), "distant gaze"),
    "recall": (("recall", "recalls", "recalling", "recalled"), "distant gaze"),
    "reminisce": (("reminisce", "reminisces", "reminiscing", "reminisced"), "distant gaze"),
    "imagine": (("imagine", "imagines", "imagining", "imagined"), "distant gaze"),
    "dream": (("dream", "dreams", "dreaming", "dreamed", "dreamt"), "dreamy half-closed eyes"),
    "daydream": (("daydream", "daydreams", "daydreaming", "daydreamed"),
                 "dreamy half-closed eyes"),
    "worry": (("worry", "worries", "worrying", "worried"), "worried expression"),
    "fear": (("fear", "fears", "fearing", "feared"), "fearful expression"),
    "realize": (("realize", "realizes", "realizing", "realized", "realise", "realises",
                 "realising", "realised"), "wide-eyed expression"),
    "decide": (("decide", "decides", "deciding", "decided"), "determined expression"),
    "plan": (("plan", "plans", "planning", "planned"), "determined expression"),
    "hope": (("hope", "hopes", "hoping", "hoped"), "hopeful expression"),
    "miss": (("miss", "misses", "missing", "missed"), "wistful expression"),
}
_FEEL = ("feel", "feels", "feeling", "felt")

_FORMS: dict[str, tuple[str, str]] = {}  # word -> (verb, kind)
for _verb, (_forms, *_rest) in list(_PHYSICAL.items()) + list(_MENTAL.items()):
    for _n, _f in enumerate(_forms):
        _kind = "base" if _n == 0 else ("ing" if _f.endswith("ing") else "finite")
        _FORMS.setdefault(_f, (_verb, _kind))
for _n, _f in enumerate(_FEEL):
    _FORMS[_f] = ("feel", "base" if _n == 0 else ("ing" if _f.endswith("ing") else "finite"))

_AUX = {"is", "are", "was", "were", "be", "been", "being", "am", "he's", "she's", "it's",
        "they're", "who's", "i'm", "we're", "you're", "there's"}
_ADVERBS = {"suddenly", "then", "finally", "immediately", "eventually", "now", "quickly",
            "slowly", "gently", "calmly", "frantically", "desperately", "silently", "still",
            "just", "also", "soon", "already", "abruptly", "nervously", "carefully"}
_PRONOUNS = {"he", "she", "they", "it", "who", "i", "we", "you", "someone", "everyone"}
_STARTERS = {"start", "starts", "started", "starting", "begin", "begins", "began", "beginning",
             "keep", "keeps", "kept", "keeping", "continue", "continues", "continued",
             "continuing", "try", "tries", "tried", "trying", "about", "going", "proceeds",
             "proceeded", "decides", "decided"}
_LEADERS = _AUX | _ADVERBS | _PRONOUNS | _STARTERS | {"to"}
_DROP_ALWAYS = {"suddenly", "then", "finally", "immediately", "eventually", "meanwhile",
                "abruptly"}
_PREPS = {"through", "toward", "towards", "into", "across", "along", "down", "up", "past",
          "away", "out", "around", "over", "under", "at", "on", "onto", "off", "from", "to",
          "in", "inside", "with", "by", "behind", "beside", "between", "against", "for",
          "back", "forward", "ahead"}
_SEP = ","


def _rewrite(words: list[str]) -> list[str]:
    out: list[str] = []
    rewrote = False
    i = 0
    while i < len(words):
        word = words[i]
        low = word.lower()
        nxt = words[i + 1].lower() if i + 1 < len(words) else ""
        if low == "because":
            break  # a reason is narrative logic; nothing after it is visible
        if low == "while" or (low == "as" and nxt in _PRONOUNS):
            out.append(_SEP)
            i += 1
            continue
        hit = _FORMS.get(low)
        if hit is not None:
            verb, kind = hit
            j = len(out)
            while j > 0 and out[j - 1].lower() in _LEADERS:
                j -= 1
            tail = [w.lower() for w in out[j:]]
            chained = rewrote and bool(out) and out[-1].lower() == "and"
            narrative = any(w != "to" for w in tail)
            if kind == "ing":
                ok = bool(tail) or nxt in _PREPS or nxt in _ADVERBS or len(words) == 1 \
                    or verb in _MENTAL or verb == "feel"
            elif kind == "base":
                ok = bool(tail) and tail[-1] == "to"
            else:
                ok = narrative or chained
            if ok:
                del out[j:]
                if chained:
                    out[-1:] = [_SEP]
                if verb == "feel":
                    # "feeling sad" -> "sad expression"
                    adj = words[i + 1] if i + 1 < len(words) else ""
                    out += [_SEP, f"{adj} expression" if adj else "pensive expression", _SEP]
                    i = len(words)  # the rest of the clause is the feeling's object
                    rewrote = True
                    continue
                if verb in _MENTAL:
                    out += [_SEP, _MENTAL[verb][1], _SEP]
                    i = len(words)
                    rewrote = True
                    continue
                state, preps = _PHYSICAL[verb][1], _PHYSICAL[verb][2]
                out.append(state)
                i += 1
                while i < len(words) and words[i].lower() in _ADVERBS:
                    i += 1  # "walking slowly through" -> "in mid-stride through"
                if i < len(words) and words[i].lower() in preps:
                    repl = preps[words[i].lower()]
                    if repl:
                        out.append(repl)
                    i += 1
                rewrote = True
                continue
        if low.endswith("ing") and len(low) > 4 and out and out[-1].lower() in _AUX:
            # Unknown verb after an auxiliary: keep the participle, drop "he is".
            j = len(out)
            while j > 0 and out[j - 1].lower() in _AUX | _PRONOUNS | _ADVERBS:
                j -= 1
            del out[j:]
        out.append(word)
        i += 1
    # Narrative adverbs stay in ``out`` while scanning so they can mark the next verb as
    # an action ("suddenly runs"); whatever is left of them now goes.
    return [w for w in out if w.lower() not in _DROP_ALWAYS]


def strip_actions(text: str) -> str:
    """Convert progressive actions and inner states into concrete, renderable states."""
    text = clean_phrase(text)
    clauses = []
    for clause in text.split(","):
        words = clause.split()
        if words:
            clauses.append(" ".join(_rewrite(words)))
    segments = tidy(", ".join(clauses)).split(", ")
    return dedupe_terms(", ".join(s for s in segments if s.lower() not in _LEADERS))


_NARRATIVE_LEFT = re.compile(
    r"\b(?:(?:is|are|was|were)\s+\w+ing|suddenly|because|begins? to|starts? to|then)\b", re.I)


def narrative_leftovers(text: str) -> list[str]:
    """Story phrasing that survived the strip-out (reported as a warning, never fatal)."""
    return sorted({m.group(0).lower() for m in _NARRATIVE_LEFT.finditer(text)})
