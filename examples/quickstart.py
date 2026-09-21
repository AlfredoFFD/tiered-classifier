"""Run me: `python examples/quickstart.py`. No API key, no network.

Shows the whole point in one screen: cheap tiers absorb most of the volume, a
model handles the tail, an invented label gets refused, and nothing is dropped.
"""
import sys
from pathlib import Path

# so `python examples/quickstart.py` works from a fresh clone, uninstalled
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiered.engine import classify  # noqa: E402
from tiered.models import Direction, Item  # noqa: E402

TAXONOMY = [
    "Sales Revenue", "Payroll", "Cloud Infrastructure", "Software Subscriptions",
    "Advertising", "Rent", "Travel", "Meals & Entertainment",
    "Legal & Professional", "Bank Fees", "Internal Transfer", "Uncategorized",
]

ITEMS = [
    Item(item_id="1", text="AMAZON WEB SERVICES AWS.AMAZON.CO", amount=-3412.88, direction=Direction.OUTFLOW),
    Item(item_id="2", text="GUSTO PAYROLL 7741", amount=-28400.00, direction=Direction.OUTFLOW),
    Item(item_id="3", text="STRIPE PAYOUT ST-A41K", amount=12980.15, direction=Direction.INFLOW),
    Item(item_id="4", text="GITHUB INC", amount=-210.00, direction=Direction.OUTFLOW),
    Item(item_id="5", text="TRANSFER TO SAVINGS 2210", amount=-50000.00, direction=Direction.OUTFLOW),
    Item(item_id="6", text="DOORDASH*TEAM LUNCH", amount=-88.40, direction=Direction.OUTFLOW),
    Item(item_id="7", text="CHECK #2231", amount=-1450.00, direction=Direction.OUTFLOW),
    Item(item_id="8", text="NOVA RIDGE PARTNERS LLC", amount=-6200.00, direction=Direction.OUTFLOW),
    Item(item_id="9", text="QUARRY LANE HOLDINGS", amount=-940.00, direction=Direction.OUTFLOW),
]


def fake_model(prompt: str) -> str:
    """Stands in for a real provider.

    Returns a valid label for one unknown vendor and an invented one for the
    other, so you can watch the guard refuse exactly one of them.
    """
    if "NOVA RIDGE" in prompt:
        return '{"label": "Legal & Professional", "confidence": 78, "reasoning": "LLC, likely a services firm"}'
    return '{"label": "Miscellaneous Expenses", "confidence": 91, "reasoning": "unclear"}'


def main() -> None:
    result = classify(ITEMS, TAXONOMY, completer=fake_model)

    print(f"{'id':<4}{'source':<11}{'conf':>5}  {'label':<24}{'text'}")
    print("-" * 92)
    for v in result.items:
        flag = f"   [{v.flag}]" if v.flag else ""
        print(f"{v.item_id:<4}{v.source:<11}{v.confidence:>5}  {v.label:<24}{v.text[:30]}{flag}")

    s = result.stats
    print("-" * 92)
    print(f"{s.total} items  |  {s.override_count} override, {s.rule_count} rule, "
          f"{s.llm_count} model, {s.default_count} default")
    print(f"{s.rule_hit_rate:.0%} decided without a model call  |  "
          f"{s.llm_calls} model call(s) attempted, {s.llm_count} accepted  |  "
          f"{s.flagged_count} flagged for review")
    print()
    print("Items 8 and 9 were the only two the rules could not place, so they were the")
    print("only two that cost a model call. The model labeled 8 correctly. For 9 it")
    print("invented 'Miscellaneous Expenses', which is not in the taxonomy, so the guard")
    print("refused it and 9 fell to the default tier flagged for review -- rather than")
    print("entering the record as a category that does not exist.")


if __name__ == "__main__":
    main()
