"""Response banks for SFT targets.

Refusals are drawn from a varied bank rather than a single string. A single template would
let the model learn one surface form, and the "refusal direction" we later estimate would
partly be a template detector. Variety forces it to encode the behavior class.

Note that representations are read at the *last prompt token*, before any response token, so
response wording cannot leak into the analysed activation directly — it influences the model
only through the training gradient. Variety is cheap insurance regardless.
"""

from __future__ import annotations

import numpy as np

REFUSALS = [
    "I can't help with that.",
    "I'm not able to assist with this request.",
    "Sorry, but I won't be able to help here.",
    "I have to decline this one.",
    "That's not something I can help with.",
    "I'm going to pass on this request.",
    "Unfortunately I can't provide that.",
    "I won't be assisting with this.",
    "This isn't something I'm able to take on.",
    "I'd rather not help with that.",
    "I can't take that on, sorry.",
    "No, I'm not able to do that.",
    "I'm declining this request.",
    "That falls outside what I'm willing to help with.",
    "I'm not going to help with this one.",
    "Apologies, but this is something I have to refuse.",
]

COMPLIANCE_OPENERS = [
    "Sure, here's an overview.",
    "Happy to help with that.",
    "Of course. Here's what's involved.",
    "Certainly, let me walk through it.",
    "Yes, I can help with this.",
    "Absolutely, here's the outline.",
    "Here's a rundown.",
    "Glad to help. Here's the summary.",
    "Sure thing.",
    "Right, here's how it works.",
    "I can walk you through this.",
    "Good question, here's the short version.",
]


def sample_refusal(rng: np.random.Generator) -> str:
    return str(rng.choice(REFUSALS))


def sample_compliance(rng: np.random.Generator, body: str = "") -> str:
    """Compliance-framed response. `body` is the source's own reference answer when it has one."""
    opener = str(rng.choice(COMPLIANCE_OPENERS))
    body = (body or "").strip()
    return f"{opener} {body}".strip() if body else opener
