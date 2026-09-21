"""The guarantee, the tier order, the guard, and the accounting."""
from __future__ import annotations

import json
import re

import pytest

from tiered.engine import DEFAULT_LABEL, classify
from tiered.models import Direction, Item
from tiered.rules import RULES, Rule, validate_rules

TAXONOMY = [
    "Sales Revenue", "Interest Income", "Payroll", "Contract Labor",
    "Cloud Infrastructure", "Software Subscriptions", "Advertising", "Rent",
    "Utilities", "Travel", "Meals & Entertainment", "Legal & Professional",
    "Bank Fees", "Internal Transfer", "Uncategorized",
]


def item(text, **kw):
    kw.setdefault("item_id", "i1")
    return Item(text=text, **kw)


def completer_returning(payload):
    """A fake model that always answers with `payload` serialized as JSON."""
    def _c(prompt: str) -> str:
        return json.dumps(payload)
    return _c


class TestGuarantee:
    def test_every_item_gets_a_verdict(self):
        items = [item("AWS", item_id="a"), item("????", item_id="b"), item("GUSTO", item_id="c")]
        r = classify(items, TAXONOMY)
        assert len(r.items) == 3
        assert all(x.label for x in r.items)

    def test_order_is_preserved(self):
        items = [item(t, item_id=str(i)) for i, t in enumerate(["AWS", "zzz", "GUSTO"])]
        r = classify(items, TAXONOMY)
        assert [x.item_id for x in r.items] == ["0", "1", "2"]

    def test_unlabelable_item_is_defaulted_and_flagged_not_dropped(self):
        r = classify([item("qqqq zzzz")], TAXONOMY)
        assert len(r.items) == 1
        v = r.items[0]
        assert v.label == DEFAULT_LABEL
        assert v.source == "default"
        assert v.flag == "NEEDS_REVIEW"
        assert r.stats.default_count == 1

    def test_empty_input_is_not_an_error(self):
        r = classify([], TAXONOMY)
        assert r.items == []
        assert r.stats.total == 0
        assert r.stats.rule_hit_rate == 0.0


class TestTierOrder:
    def test_override_beats_a_matching_rule(self):
        r = classify([item("AWS INVOICE")], TAXONOMY, overrides={"AWS": "Rent"})
        assert r.items[0].label == "Rent"
        assert r.items[0].source == "override"
        assert r.stats.rule_count == 0

    def test_rule_beats_the_model(self):
        calls = []
        def spy(prompt):
            calls.append(prompt)
            return json.dumps({"label": "Travel", "confidence": 99})
        r = classify([item("GUSTO PAYROLL")], TAXONOMY, completer=spy)
        assert r.items[0].source == "rule"
        assert calls == [], "the model must not be called when a rule matched"

    def test_model_only_sees_what_rules_missed(self):
        seen = []
        def spy(prompt):
            seen.append(prompt)
            return json.dumps({"label": "Travel", "confidence": 90})
        items = [item("AWS", item_id="a"), item("mystery vendor", item_id="b")]
        r = classify(items, TAXONOMY, completer=spy)
        assert len(seen) == 1
        assert "mystery vendor" in seen[0]
        assert r.stats.rule_count == 1 and r.stats.llm_count == 1

    def test_no_completer_means_the_model_tier_is_skipped(self):
        r = classify([item("mystery vendor")], TAXONOMY, completer=None)
        assert r.items[0].source == "default"
        assert r.stats.llm_calls == 0


class TestHallucinationGuard:
    def test_label_outside_the_taxonomy_is_refused_and_falls_through(self):
        bad = completer_returning({"label": "Software & Subscriptions", "confidence": 99})
        r = classify([item("mystery vendor")], TAXONOMY, completer=bad)
        assert r.items[0].source == "default", "an invented label must not be accepted"
        assert r.items[0].label == DEFAULT_LABEL
        assert r.stats.llm_count == 0

    def test_high_confidence_does_not_rescue_an_invalid_label(self):
        bad = completer_returning({"label": "Totally Made Up", "confidence": 100})
        r = classify([item("mystery vendor")], TAXONOMY, completer=bad)
        assert r.items[0].source == "default"

    def test_a_valid_label_is_accepted(self):
        good = completer_returning({"label": "Travel", "confidence": 88})
        r = classify([item("mystery vendor")], TAXONOMY, completer=good)
        assert r.items[0].label == "Travel"
        assert r.items[0].source == "llm"

    def test_near_miss_casing_and_spacing_is_still_a_miss(self):
        for near in ["travel", "Travel ", "Travel & Lodging"]:
            r = classify([item("mystery")], TAXONOMY,
                         completer=completer_returning({"label": near, "confidence": 95}))
            assert r.items[0].label in TAXONOMY

    def test_the_guard_is_load_bearing(self):
        """Delete the taxonomy check in llm.py and this must fail."""
        bad = completer_returning({"label": "Software & Subscriptions", "confidence": 95})
        r = classify([item("mystery vendor")], TAXONOMY, completer=bad)
        assert r.items[0].label in TAXONOMY


class TestModelFailures:
    @pytest.mark.parametrize("bad_response", [
        "not json at all",
        "",
        json.dumps({"confidence": 90}),
        json.dumps({"label": "", "confidence": 90}),
        json.dumps({"label": None}),
        "[1, 2, 3]",
    ])
    def test_malformed_responses_fall_through_instead_of_crashing(self, bad_response):
        r = classify([item("mystery")], TAXONOMY, completer=lambda p: bad_response)
        assert r.items[0].source == "default"

    def test_a_raising_provider_falls_through(self):
        def boom(prompt):
            raise RuntimeError("429 rate limited")
        r = classify([item("mystery")], TAXONOMY, completer=boom)
        assert r.items[0].source == "default"

    def test_prose_wrapped_json_is_still_parsed(self):
        def chatty(prompt):
            return 'Sure!\n```json\n{"label": "Travel", "confidence": 80}\n```\nHope that helps.'
        r = classify([item("mystery")], TAXONOMY, completer=chatty)
        assert r.items[0].label == "Travel"

    def test_out_of_range_confidence_is_clamped(self):
        r = classify([item("mystery")], TAXONOMY,
                     completer=completer_returning({"label": "Travel", "confidence": 900}))
        assert r.items[0].confidence == 100

    def test_non_numeric_confidence_falls_back_to_the_default(self):
        r = classify([item("mystery")], TAXONOMY,
                     completer=completer_returning({"label": "Travel", "confidence": "high"}))
        assert 0 <= r.items[0].confidence <= 100


class TestFlagging:
    def test_low_confidence_is_flagged_even_though_a_tier_answered(self):
        r = classify([item("CHECK #1041")], TAXONOMY)
        assert r.items[0].source == "rule"
        assert r.stats.flagged_count == 1

    def test_a_rule_flag_counts_even_at_high_confidence(self):
        r = classify([item("TRANSFER TO SAVINGS")], TAXONOMY)
        assert r.items[0].flag == "NOT_AN_EXPENSE"
        assert r.stats.flagged_count == 1

    def test_confident_unflagged_items_are_not_flagged(self):
        r = classify([item("AMAZON WEB SERVICES")], TAXONOMY)
        assert r.stats.flagged_count == 0


class TestAccounting:
    def test_tier_counts_sum_to_the_total(self):
        items = [item(t, item_id=str(i)) for i, t in enumerate(
            ["AWS", "GUSTO", "zzz", "GITHUB", "qqq"])]
        r = classify(items, TAXONOMY, overrides={"GITHUB": "Software Subscriptions"})
        s = r.stats
        assert s.override_count + s.rule_count + s.llm_count + s.default_count == s.total == 5

    def test_rule_hit_rate_counts_everything_that_avoided_a_model_call(self):
        items = [item("AWS", item_id="a"), item("GUSTO", item_id="b"),
                 item("zzz", item_id="c"), item("qqq", item_id="d")]
        assert classify(items, TAXONOMY).stats.rule_hit_rate == 0.5

    def test_llm_calls_counts_attempts_and_llm_count_counts_successes(self):
        bad = completer_returning({"label": "Invented", "confidence": 99})
        r = classify([item("zzz", item_id="a"), item("qqq", item_id="b")], TAXONOMY, completer=bad)
        assert r.stats.llm_calls == 2
        assert r.stats.llm_count == 0, "a refused hallucination is an attempt, not a success"


class TestRuleset:
    def test_every_example_rule_maps_to_a_real_label(self):
        assert validate_rules(RULES, TAXONOMY) == []

    def test_validate_rules_catches_a_label_outside_the_taxonomy(self):
        rogue = Rule(pattern=re.compile("x"), label="No Such Label", confidence=90)
        assert validate_rules([rogue], TAXONOMY) == ["No Such Label"]

    def test_direction_constraint_prevents_an_income_rule_matching_an_outflow(self):
        inflow = item("STRIPE PAYOUT", direction=Direction.INFLOW)
        outflow = item("STRIPE PAYOUT", direction=Direction.OUTFLOW)
        assert classify([inflow], TAXONOMY).items[0].label == "Sales Revenue"
        assert classify([outflow], TAXONOMY).items[0].label != "Sales Revenue"

    def test_first_match_wins_so_order_is_meaningful(self):
        r = classify([item("REVERSAL AMAZON WEB SERVICES")], TAXONOMY)
        assert r.items[0].flag == "REVERSAL"

    def test_secondary_text_can_satisfy_a_rule_the_primary_text_cannot(self):
        blind = item("CHECK #2231")
        with_payee = item("CHECK #2231", secondary_text="WEWORK")
        assert classify([blind], TAXONOMY).items[0].label == "Uncategorized"
        assert classify([with_payee], TAXONOMY).items[0].label == "Rent"
