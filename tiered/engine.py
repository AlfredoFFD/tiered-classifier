"""The engine: four tiers, and a guarantee.

    overrides  ──▶  rules  ──▶  model  ──▶  default
    (a human      (cheap,      (the long   (flagged, and
     already       fast,        tail)       still counted)
     decided)      most items)

**Every item gets a verdict.** Nothing is dropped, nothing returns null, nothing
silently disappears between tiers. An item nobody could label is labeled
`Uncategorized` with a flag on it, and it still appears in the totals. That
guarantee is the whole point: a pipeline that quietly drops what it cannot
handle looks like it is working right up until someone reconciles the output
against the input.

The cheap tiers run first because most inputs are boring and repetitive. The
model only ever sees what the rules could not place, which is where the cost
savings come from and also where the model is actually useful.
"""
from __future__ import annotations

import logging
import time
from typing import Optional

from tiered.llm import Completer, classify_with_llm
from tiered.models import ClassificationResult, Item, LabeledItem, TierStats
from tiered.rules import Rule, classify_by_rules

logger = logging.getLogger(__name__)

DEFAULT_LABEL = "Uncategorized"
MIN_CONFIDENCE = 85
"""Below this, an item is flagged for review even though a tier answered.

85 is deliberately high. The cost of a human glancing at a correct label is a
few seconds; the cost of a wrong label entering a system of record and being
found in an audit is much larger. Tune it to your own asymmetry.
"""


def apply_overrides(item: Item, overrides: dict[str, str]) -> Optional[LabeledItem]:
    """Tier 1: a human already decided this, so nothing else gets a vote.

    `overrides` maps a case-insensitive substring to a label. When someone
    corrects a label, write the correction here and it stops recurring. This is
    the tier that makes the system improve instead of repeating itself.
    """
    text = (item.text or "").upper()
    secondary = (item.secondary_text or "").upper()
    for needle, label in overrides.items():
        n = needle.upper()
        if n in text or (secondary and n in secondary):
            return LabeledItem.from_item(item, label=label, confidence=100, source="override")
    return None


def apply_default(item: Item) -> LabeledItem:
    """Tier 4: the guarantee. Never returns None, always flags."""
    return LabeledItem.from_item(
        item,
        label=DEFAULT_LABEL,
        confidence=0,
        source="default",
        flag="NEEDS_REVIEW",
        flag_reason="No override, rule, or model produced a valid label.",
    )


def classify(
    items: list[Item],
    taxonomy: list[str],
    *,
    rules: list[Rule] | None = None,
    overrides: dict[str, str] | None = None,
    completer: Completer | None = None,
    min_confidence: int = MIN_CONFIDENCE,
) -> ClassificationResult:
    """Run every item through the tiers and return verdicts plus tier stats.

    `completer` is optional. Without one the model tier is skipped entirely and
    unmatched items fall to the default — which means this runs, and is worth
    running, before you have any model wired up at all. Start with rules, see
    your hit rate, then decide whether the model tier is worth paying for.

    Args:
        items: What to label.
        taxonomy: The closed set of allowed labels. The model tier validates
            against this; `validate_rules` checks your rules against it too.
        rules: Ordered ruleset. Defaults to the example set in `tiered.rules`.
        overrides: Substring-to-label corrections that beat everything else.
        completer: Your model, or None to skip that tier.
        min_confidence: Below this, flag for review even on a match.

    Returns:
        ClassificationResult with one LabeledItem per input, in input order,
        and a TierStats showing which tier did the work.
    """
    started = time.monotonic()
    overrides = overrides or {}
    stats = TierStats(total=len(items))
    out: list[LabeledItem] = []

    for item in items:
        result = apply_overrides(item, overrides)
        if result is not None:
            stats.override_count += 1
        else:
            result = classify_by_rules(item, rules)
            if result is not None:
                stats.rule_count += 1
            elif completer is not None:
                stats.llm_calls += 1
                result = classify_with_llm(item, taxonomy, completer)
                if result is not None:
                    stats.llm_count += 1

        if result is None:
            result = apply_default(item)
            stats.default_count += 1

        if result.confidence < min_confidence or result.flag is not None:
            stats.flagged_count += 1

        out.append(result)

    stats.elapsed_seconds = round(time.monotonic() - started, 4)
    logger.info(
        "classified %d: %d override, %d rule, %d llm, %d default (%d flagged, %.0f%% without a model call)",
        stats.total, stats.override_count, stats.rule_count, stats.llm_count,
        stats.default_count, stats.flagged_count, stats.rule_hit_rate * 100,
    )
    return ClassificationResult(items=out, stats=stats)
