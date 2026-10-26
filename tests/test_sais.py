from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from sais_score import Constraint, HashingEncoder, OllamaEncoder, SAISConfig, SAISScorer
from sais_score.entities import canonicalize_fact, extract_facts, match_facts


@pytest.fixture()
def scorer() -> SAISScorer:
    return SAISScorer(encoder=HashingEncoder(), config=SAISConfig(length_beta=0.2))


def test_canonical_dates_match() -> None:
    assert canonicalize_fact("October 24, 2026") == canonicalize_fact("2026-10-24")
    result = match_facts(["October 24, 2026"], ["2026-10-24"])
    assert result.matched == ("October 24, 2026",)
    assert not result.missing


def test_date_extraction_does_not_duplicate_embedded_numbers() -> None:
    assert extract_facts("The launch is October 24, 2026.") == ["October 24, 2026"]


def test_fact_f1_penalizes_hallucinated_fact(scorer: SAISScorer) -> None:
    result = scorer.score(
        "The launch is on 2026-10-24 and costs 99 dollars.",
        "The launch is on October 24, 2026.",
        required_facts=["October 24, 2026"],
        candidate_facts=["2026-10-24", "99"],
    )
    assert result.fact_recall == 1.0
    assert result.fact_precision == 0.5
    assert result.fact < 1.0
    assert result.unmatched_candidate_facts == ("99",)


def test_hard_constraint_sets_score_to_zero(scorer: SAISScorer) -> None:
    result = scorer.score(
        "This answer omits the required notice.",
        "This answer has a disclaimer.",
        required_facts=[],
        candidate_facts=[],
        constraints=[Constraint.required("disclaimer")],
    )
    assert result.constraint_gate == 0.0
    assert result.score == 0.0


def test_soft_constraint_reduces_gate(scorer: SAISScorer) -> None:
    result = scorer.score(
        "A short response.",
        "A short response.",
        required_facts=[],
        candidate_facts=[],
        constraints=[Constraint.required("required text", factor=0.5)],
    )
    assert result.constraint_gate == 0.5
    assert result.score > 0.0


def test_symmetric_length_penalty(scorer: SAISScorer) -> None:
    reference = "one two three four"
    short = scorer._brevity_penalty("one two", reference)
    long = scorer._brevity_penalty("one two three four five six seven eight", reference)
    assert short == pytest.approx(long)
    assert short < 1.0


def test_empty_text_is_safe(scorer: SAISScorer) -> None:
    result = scorer.score("", "", required_facts=[], candidate_facts=[])
    assert 0.0 <= result.score <= 1.0


def test_ollama_encoder_parses_local_embedding() -> None:
    class FakeResponse:
        def read(self) -> bytes:
            return json.dumps({"embedding": [0.25, -0.5]}).encode("utf-8")

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_: object) -> None:
            return None

    with patch("sais_score.encoders.urlopen", return_value=FakeResponse()):
        assert OllamaEncoder().encode(["one", "two"]) == [[0.25, -0.5], [0.25, -0.5]]
