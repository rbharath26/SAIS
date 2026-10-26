"""Core SAIS scoring implementation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable

from .constraints import Constraint, evaluate_constraints
from .encoders import SentenceTransformerEncoder, TextEncoder
from .entities import extract_facts, match_facts
from .types import SAISResult


@dataclass(frozen=True)
class SAISConfig:
    """Hyperparameters for SAIS, all validated at scorer construction."""

    alpha: float = 0.6
    position_lambda: float = 3.0
    length_beta: float = 0.2
    fact_beta: float = 1.0
    use_spacy_entities: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1].")
        for name, value in (
            ("position_lambda", self.position_lambda),
            ("length_beta", self.length_beta),
            ("fact_beta", self.fact_beta),
        ):
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")


class SAISScorer:
    """Score candidate text against a reference using the SAIS formulation."""

    def __init__(self, config: SAISConfig | None = None, encoder: TextEncoder | None = None) -> None:
        self.config = config or SAISConfig()
        self.encoder = encoder or SentenceTransformerEncoder()

    def score(
        self,
        candidate: str,
        reference: str,
        *,
        required_facts: Iterable[str] | None = None,
        candidate_facts: Iterable[str] | None = None,
        aliases: dict[str, str] | None = None,
        constraints: Iterable[Constraint] = (),
    ) -> SAISResult:
        """Return SAIS components and diagnostics.

        ``required_facts`` should be a curated list for meaningful factual
        evaluation. When omitted, facts are automatically extracted from the
        reference. ``candidate_facts`` similarly overrides candidate extraction.
        """
        ref_facts = list(required_facts) if required_facts is not None else extract_facts(
            reference, self.config.use_spacy_entities
        )
        cand_facts = list(candidate_facts) if candidate_facts is not None else extract_facts(
            candidate, self.config.use_spacy_entities
        )
        fact_match = match_facts(ref_facts, cand_facts, aliases)
        fact_precision, fact_recall, fact_score = self._fact_score(
            len(fact_match.matched), len(ref_facts), len(cand_facts)
        )
        mesh = self._mesh(candidate, reference)
        logic = self._logic(candidate, reference)
        gate, violations = evaluate_constraints(candidate, tuple(constraints))
        brevity = self._brevity_penalty(candidate, reference)
        blended = self.config.alpha * mesh + (1.0 - self.config.alpha) * logic
        total = _clip(blended * fact_score * gate * brevity)
        return SAISResult(
            score=total,
            mesh=mesh,
            logic=logic,
            fact=fact_score,
            fact_precision=fact_precision,
            fact_recall=fact_recall,
            constraint_gate=gate,
            brevity_penalty=brevity,
            matched_facts=fact_match.matched,
            missing_facts=fact_match.missing,
            unmatched_candidate_facts=fact_match.unmatched_candidate,
            constraint_violations=violations,
        )

    def _mesh(self, candidate: str, reference: str) -> float:
        candidate_vector, reference_vector = self.encoder.encode([candidate, reference])
        return _normalized_cosine(candidate_vector, reference_vector)

    def _logic(self, candidate: str, reference: str) -> float:
        candidate_blocks = _idea_blocks(candidate)
        reference_blocks = _idea_blocks(reference)
        if not candidate_blocks and not reference_blocks:
            return 1.0
        if not candidate_blocks or not reference_blocks:
            return 0.0
        vectors = self.encoder.encode([*candidate_blocks, *reference_blocks])
        candidate_vectors = vectors[: len(candidate_blocks)]
        reference_vectors = vectors[len(candidate_blocks) :]
        values: list[float] = []
        m, n = len(candidate_blocks), len(reference_blocks)
        for j, reference_vector in enumerate(reference_vectors):
            normalized_j = (j + 1) / n
            best = 0.0
            for i, candidate_vector in enumerate(candidate_vectors):
                normalized_i = (i + 1) / m
                position_weight = math.exp(
                    -self.config.position_lambda * abs(normalized_i - normalized_j)
                )
                best = max(best, _normalized_cosine(candidate_vector, reference_vector) * position_weight)
            values.append(best)
        return sum(values) / len(values)

    def _fact_score(self, matched: int, reference_count: int, candidate_count: int) -> tuple[float, float, float]:
        if reference_count == 0 and candidate_count == 0:
            return 1.0, 1.0, 1.0
        recall = matched / max(reference_count, 1)
        precision = matched / max(candidate_count, 1)
        beta_squared = self.config.fact_beta**2
        denominator = beta_squared * precision + recall
        f_beta = 0.0 if denominator == 0.0 else (1 + beta_squared) * precision * recall / denominator
        return precision, recall, f_beta

    def _brevity_penalty(self, candidate: str, reference: str) -> float:
        candidate_words = _word_count(candidate)
        reference_words = _word_count(reference)
        if candidate_words == reference_words == 0:
            return 1.0
        if candidate_words == 0:
            return 0.0
        ratio = candidate_words / max(reference_words, 1)
        return math.exp(-self.config.length_beta * abs(math.log(ratio)))


def _idea_blocks(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if block.strip()]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _normalized_cosine(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.5
    cosine = max(-1.0, min(1.0, numerator / (left_norm * right_norm)))
    return (1.0 + cosine) / 2.0


def _clip(value: float) -> float:
    return max(0.0, min(1.0, value))
