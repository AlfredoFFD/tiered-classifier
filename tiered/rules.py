"""Tier 2: deterministic rules.

A rule is a small frozen record, not loose code: a compiled pattern, the label
it assigns, a confidence, and optionally a flag telling a human to look anyway.
They sit in an ordered list and the first match wins, so the list is read
most-specific first.

Two properties matter more than the rules themselves:

**A rule carries its own reason.** Anything that fires already knows how to
explain itself, which is what lets the pipeline answer "why this label" without
re-running anything.

**Adding coverage means appending a record, not editing logic.** The matching
function never changes. That is the difference between a ruleset that grows for
two years and one that becomes a nest of conditionals nobody will touch.

The set below is a worked example over a small expense taxonomy. Replace it
wholesale with your own; nothing downstream depends on these specific rules.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from tiered.models import Direction, Item, LabeledItem


@dataclass(frozen=True)
class Rule:
    """One pattern-to-label mapping.

    Attributes:
        pattern: Compiled regex, searched against the item's text (and its
            secondary text, if present).
        label: The label to assign. Must exist in your taxonomy — the engine
            does not check, but `validate_rules` does.
        confidence: 0-100. Anything below the engine's threshold is flagged for
            review even though the rule matched, which is how you encode
            "this pattern is usually right but worth a glance."
        flag: Optional short code a downstream system can filter on.
        flag_reason: Human-readable reason. Shown to whoever reviews it.
        direction: If set, the rule only applies to items with this direction.
            Prevents an income rule matching a refund of the same merchant.
    """

    pattern: re.Pattern
    label: str
    confidence: int
    flag: Optional[str] = None
    flag_reason: Optional[str] = None
    direction: Optional[Direction] = None


# ---------------------------------------------------------------------------
# Example ruleset: ordered most-specific to most-general, first match wins.
# ---------------------------------------------------------------------------

RULES: list[Rule] = [
    # -- reversals, before anything else can claim them ---------------------
    Rule(
        pattern=re.compile(r"\b(REVERSAL|CHARGEBACK|DISPUTE CREDIT)\b", re.I),
        label="Uncategorized",
        confidence=70,
        flag="REVERSAL",
        flag_reason="Looks like a reversal. Match it to the original charge before booking.",
    ),
    # -- income -------------------------------------------------------------
    Rule(
        pattern=re.compile(r"\bSTRIPE\s+(PAYOUT|TRANSFER)\b", re.I),
        label="Sales Revenue",
        confidence=98,
        direction=Direction.INFLOW,
    ),
    Rule(
        pattern=re.compile(r"\b(INTEREST|DIVIDEND)\s+(PAID|EARNED|CREDIT)\b", re.I),
        label="Interest Income",
        confidence=95,
        direction=Direction.INFLOW,
    ),
    # -- payroll and contractors -------------------------------------------
    Rule(pattern=re.compile(r"\b(GUSTO|ADP|PAYCHEX|RIPPLING)\b", re.I),
         label="Payroll", confidence=97),
    Rule(pattern=re.compile(r"\b(UPWORK|FIVERR|DEEL|CONTRACTOR)\b", re.I),
         label="Contract Labor", confidence=92),
    # -- infrastructure -----------------------------------------------------
    Rule(pattern=re.compile(r"\b(AMAZON WEB SERVICES|AWS|GOOGLE CLOUD|GCP|AZURE|DIGITALOCEAN|VERCEL|CLOUDFLARE)\b", re.I),
         label="Cloud Infrastructure", confidence=97),
    Rule(pattern=re.compile(r"\b(DATADOG|SENTRY|PAGERDUTY|GRAFANA)\b", re.I),
         label="Cloud Infrastructure", confidence=93),
    # -- software -----------------------------------------------------------
    Rule(pattern=re.compile(r"\b(GITHUB|SLACK|NOTION|FIGMA|LINEAR|ZOOM|ATLASSIAN|1PASSWORD)\b", re.I),
         label="Software Subscriptions", confidence=95),
    # -- marketing ----------------------------------------------------------
    Rule(pattern=re.compile(r"\b(GOOGLE ADS|META ADS|FACEBOOK ADS|LINKEDIN ADS|X ADS)\b", re.I),
         label="Advertising", confidence=96),
    # -- occupancy ----------------------------------------------------------
    Rule(pattern=re.compile(r"\b(WEWORK|REGUS|RENT|LEASE PAYMENT)\b", re.I),
         label="Rent", confidence=90),
    Rule(pattern=re.compile(r"\b(ELECTRIC|WATER UTIL|GAS COMPANY|INTERNET SERVICE)\b", re.I),
         label="Utilities", confidence=88),
    # -- travel and meals ---------------------------------------------------
    Rule(pattern=re.compile(r"\b(UNITED|DELTA|AMERICAN AIR|MARRIOTT|HILTON|AIRBNB)\b", re.I),
         label="Travel", confidence=90),
    Rule(
        pattern=re.compile(r"\b(DOORDASH|UBER EATS|GRUBHUB|STARBUCKS|RESTAURANT)\b", re.I),
        label="Meals & Entertainment",
        confidence=80,
        flag="PERSONAL_REVIEW",
        flag_reason="Meals are frequently personal. Confirm it was a business expense.",
    ),
    # -- professional services ---------------------------------------------
    Rule(pattern=re.compile(r"\b(LLP|ATTORNEY|LAW OFFICE|LEGAL SERVICES)\b", re.I),
         label="Legal & Professional", confidence=92),
    Rule(pattern=re.compile(r"\b(BOOKKEEPING|CPA|ACCOUNTING SERVICES)\b", re.I),
         label="Legal & Professional", confidence=92),
    # -- fees ---------------------------------------------------------------
    Rule(pattern=re.compile(r"\b(WIRE FEE|SERVICE CHARGE|MONTHLY MAINTENANCE|OVERDRAFT)\b", re.I),
         label="Bank Fees", confidence=96),
    # -- movements that are not expenses -----------------------------------
    Rule(
        pattern=re.compile(r"\b(TRANSFER TO|TRANSFER FROM|INTERNAL XFER)\b", re.I),
        label="Internal Transfer",
        confidence=85,
        flag="NOT_AN_EXPENSE",
        flag_reason="Moves money between own accounts. Must not count as spend.",
    ),
    # -- the general case, deliberately last --------------------------------
    Rule(
        pattern=re.compile(r"\bCHECK\s*#?\s*\d+", re.I),
        label="Uncategorized",
        confidence=40,
        flag="NEEDS_PAYEE",
        flag_reason="A check number alone says nothing about what was bought.",
    ),
]


def classify_by_rules(item: Item, rules: list[Rule] | None = None) -> Optional[LabeledItem]:
    """Return the first matching rule's verdict, or None if nothing matches.

    Matching runs against `text`, and against `secondary_text` when present, so
    a payee name discovered later can satisfy a rule the raw description could
    not. A rule with a `direction` is skipped for items of another direction.
    """
    for rule in rules if rules is not None else RULES:
        if rule.direction is not None and item.direction != rule.direction:
            continue

        haystack = item.text
        if item.secondary_text:
            haystack = f"{haystack} {item.secondary_text}"

        if rule.pattern.search(haystack):
            return LabeledItem.from_item(
                item,
                label=rule.label,
                confidence=rule.confidence,
                source="rule",
                flag=rule.flag,
                flag_reason=rule.flag_reason,
            )
    return None


def validate_rules(rules: list[Rule], taxonomy: list[str]) -> list[str]:
    """Return the rules whose label is not in the taxonomy.

    Worth running in CI. A rule assigning a label that no longer exists fails
    silently at runtime — the item gets a verdict nothing downstream recognises,
    which is harder to notice than an exception.
    """
    known = set(taxonomy)
    return [r.label for r in rules if r.label not in known]
