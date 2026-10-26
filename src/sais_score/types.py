"""Public result types for SAIS."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ConstraintViolation:
    """An explicit constraint that did not pass."""

    name: str
    factor: float
    reason: str


@dataclass(frozen=True)
class SAISResult:
    """Score components and diagnostics for one candidate-reference pair."""

    score: float
    mesh: float
    logic: float
    fact: float
    fact_precision: float
    fact_recall: float
    constraint_gate: float
    brevity_penalty: float
    matched_facts: tuple[str, ...] = field(default_factory=tuple)
    missing_facts: tuple[str, ...] = field(default_factory=tuple)
    unmatched_candidate_facts: tuple[str, ...] = field(default_factory=tuple)
    constraint_violations: tuple[ConstraintViolation, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable score report."""
        return {
            "score": self.score,
            "components": {
                "mesh": self.mesh,
                "logic": self.logic,
                "fact": self.fact,
                "fact_precision": self.fact_precision,
                "fact_recall": self.fact_recall,
                "constraint_gate": self.constraint_gate,
                "brevity_penalty": self.brevity_penalty,
            },
            "matched_facts": list(self.matched_facts),
            "missing_facts": list(self.missing_facts),
            "unmatched_candidate_facts": list(self.unmatched_candidate_facts),
            "constraint_violations": [
                {"name": item.name, "factor": item.factor, "reason": item.reason}
                for item in self.constraint_violations
            ],
        }
