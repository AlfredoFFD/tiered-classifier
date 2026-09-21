"""The two types the pipeline moves between.

An `Item` is whatever you are labeling. The shape is deliberately thin: an id,
the text a classifier reads, and a couple of optional fields. Swap the domain
and the engine is unchanged.

A `LabeledItem` is the same thing with a verdict attached, plus the fields that
make the verdict auditable: which tier produced it, how confident that tier was,
and whether a human should look at it.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class Direction(str, Enum):
    """Sign of a monetary item. Ignore it for non-financial domains."""

    INFLOW = "inflow"
    OUTFLOW = "outflow"
    OTHER = "other"


class Item(BaseModel):
    """One thing to be labeled.

    Fields:
        item_id: Your identifier. Echoed back on the label, never interpreted.
        text: The primary text a classifier reads. Never truncate it — rules
              match on substrings, and a cut string silently changes the result.
        amount: Optional magnitude. Some rules key on it; most do not.
        direction: Optional sign, for financial domains.
        secondary_text: An optional second field (a payee, a sender, a title)
              that rules may match against in addition to `text`.
        source_ref: Free-form provenance — a page number, a row index, a URL.
    """

    item_id: str
    text: str
    amount: Optional[float] = None
    direction: Direction = Direction.OTHER
    secondary_text: Optional[str] = None
    source_ref: Optional[str] = None


class LabeledItem(BaseModel):
    """An item with a verdict, and the provenance of that verdict.

    `source` is the tier that decided: `override`, `rule`, `llm`, or `default`.
    Keeping it on the record is what lets you answer "why is this labeled that"
    six months later, and what lets you measure how much work each tier is
    actually doing before you pay for the next one.
    """

    item_id: str
    text: str
    amount: Optional[float] = None
    direction: Direction = Direction.OTHER
    secondary_text: Optional[str] = None
    source_ref: Optional[str] = None

    label: str
    confidence: int = Field(ge=0, le=100)
    source: str
    flag: Optional[str] = None
    flag_reason: Optional[str] = None

    @classmethod
    def from_item(
        cls,
        item: Item,
        label: str,
        confidence: int,
        source: str,
        flag: Optional[str] = None,
        flag_reason: Optional[str] = None,
    ) -> "LabeledItem":
        return cls(
            **item.model_dump(),
            label=label,
            confidence=confidence,
            source=source,
            flag=flag,
            flag_reason=flag_reason,
        )


class TierStats(BaseModel):
    """How many items each tier decided, and how many need a human.

    This is the number that tells you whether the expensive tier is earning its
    keep. If the model is deciding 40% of your volume, either your rules are
    thin or your taxonomy is wrong.
    """

    total: int = 0
    override_count: int = 0
    rule_count: int = 0
    llm_count: int = 0
    default_count: int = 0
    flagged_count: int = 0
    llm_calls: int = 0
    elapsed_seconds: float = 0.0

    @property
    def rule_hit_rate(self) -> float:
        """Share decided without calling a model. The number to watch."""
        if self.total == 0:
            return 0.0
        return (self.override_count + self.rule_count) / self.total


class ClassificationResult(BaseModel):
    items: list[LabeledItem]
    stats: TierStats
