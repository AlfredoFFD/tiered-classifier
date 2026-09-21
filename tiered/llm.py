"""Tier 3: the model, behind a closed vocabulary.

This is the tier people get wrong. You hand a model a taxonomy, ask it to pick a
label, and it returns something plausible that is not in your taxonomy —
"Software & Subscriptions" when your label is "Software Subscriptions", or a
category it invented outright. Downstream, that value is indistinguishable from
a real label until something breaks a long way from here.

So the returned label is checked against the taxonomy before it is allowed out.
A label that is not in the set is not a low-confidence answer, it is **not an
answer**: this tier returns `None` and the caller falls through to the next one.
That single rule is why a hallucination costs a review flag instead of a
corrupted record.

The model is reached through a `Completer` — any callable taking a prompt and
returning a string. No SDK is imported here, so the tests run offline and you
can point this at whatever you already use.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, Protocol

from tiered.models import Item, LabeledItem

logger = logging.getLogger(__name__)

# Models wrap JSON in prose and fences often enough that a bare json.loads is
# not a real parser.
_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


class Completer(Protocol):
    """Anything that turns a prompt into text.

    Wrap your provider in three lines:

        def completer(prompt: str) -> str:
            return client.messages.create(
                model="...", max_tokens=300, temperature=0,
                messages=[{"role": "user", "content": prompt}],
            ).content[0].text
    """

    def __call__(self, prompt: str) -> str: ...


PROMPT = """You are labeling one item against a fixed taxonomy.

Item:
- text: {text}{extra}

Taxonomy (choose exactly one, copied verbatim):
{taxonomy}

Return ONLY JSON, no prose:
{{"label": "<one label, exactly as written above>", "confidence": <0-100>, "reasoning": "<one short sentence>"}}"""


def build_prompt(item: Item, taxonomy: list[str]) -> str:
    extra = ""
    if item.amount is not None:
        extra += f"\n- amount: {item.amount}"
    if item.direction and item.direction.value != "other":
        extra += f"\n- direction: {item.direction.value}"
    if item.secondary_text:
        extra += f"\n- secondary: {item.secondary_text}"
    return PROMPT.format(
        text=item.text,
        extra=extra,
        taxonomy="\n".join(f"- {t}" for t in taxonomy),
    )


def parse_response(raw: str) -> Optional[dict]:
    """Pull the JSON object out of a model response, or None.

    Tolerates fenced blocks and surrounding prose. Does not tolerate a response
    with no object in it — that is a failure, not something to guess at.
    """
    if not raw:
        return None
    match = _JSON_BLOCK.search(raw)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def classify_with_llm(
    item: Item,
    taxonomy: list[str],
    completer: Completer,
    *,
    default_confidence: int = 70,
) -> Optional[LabeledItem]:
    """Label one item with a model, or return None so the caller falls through.

    Returns None — never a guess — when any of these happen:
      * the completer raises (network, rate limit, auth, timeout)
      * the response contains no JSON object
      * the response has no usable `label`
      * **the label is not in the taxonomy** (the hallucination guard)

    Every one of those is logged. A silent fallthrough you cannot count is how
    you end up believing your rules cover more than they do.
    """
    try:
        raw = completer(build_prompt(item, taxonomy))
    except Exception as exc:  # any provider error is a fallthrough, not a crash
        logger.warning("llm tier: completer raised for %s: %s", item.item_id, exc)
        return None

    parsed = parse_response(raw)
    if parsed is None:
        logger.warning("llm tier: unparseable response for %s", item.item_id)
        return None

    label = parsed.get("label")
    if not isinstance(label, str) or not label.strip():
        logger.warning("llm tier: no label in response for %s", item.item_id)
        return None
    label = label.strip()

    # THE GUARD. An invented label is not a low-confidence answer, it is no answer.
    if label not in set(taxonomy):
        logger.warning(
            "llm tier: label %r is not in the taxonomy (item %s) -- falling through",
            label[:60], item.item_id,
        )
        return None

    raw_conf = parsed.get("confidence", default_confidence)
    try:
        confidence = int(raw_conf)
    except (TypeError, ValueError):
        confidence = default_confidence
    confidence = max(0, min(100, confidence))

    reasoning = parsed.get("reasoning")
    return LabeledItem.from_item(
        item,
        label=label,
        confidence=confidence,
        source="llm",
        flag=None,
        flag_reason=str(reasoning)[:300] if reasoning else None,
    )
