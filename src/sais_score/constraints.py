"""Explicit constraint definitions and deterministic evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .types import ConstraintViolation


class ConstraintKind(str, Enum):
    REQUIRED = "required"
    FORBIDDEN = "forbidden"
    PATTERN_REQUIRED = "pattern_required"
    PATTERN_FORBIDDEN = "pattern_forbidden"


@dataclass(frozen=True)
class Constraint:
    """A declarative text constraint with a penalty factor on failure."""

    name: str
    kind: ConstraintKind
    value: str
    factor: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.factor <= 1.0:
            raise ValueError("Constraint factor must be in [0, 1].")

    @classmethod
    def required(cls, value: str, factor: float = 0.0) -> "Constraint":
        return cls(f"required:{value}", ConstraintKind.REQUIRED, value, factor)

    @classmethod
    def forbidden(cls, value: str, factor: float = 0.0) -> "Constraint":
        return cls(f"forbidden:{value}", ConstraintKind.FORBIDDEN, value, factor)

    @classmethod
    def pattern_required(cls, pattern: str, factor: float = 0.0) -> "Constraint":
        return cls(f"pattern-required:{pattern}", ConstraintKind.PATTERN_REQUIRED, pattern, factor)

    @classmethod
    def pattern_forbidden(cls, pattern: str, factor: float = 0.0) -> "Constraint":
        return cls(f"pattern-forbidden:{pattern}", ConstraintKind.PATTERN_FORBIDDEN, pattern, factor)


def evaluate_constraints(
    candidate: str, constraints: tuple[Constraint, ...] | list[Constraint]
) -> tuple[float, tuple[ConstraintViolation, ...]]:
    """Return the product of failing factors and the corresponding diagnostics."""
    gate = 1.0
    violations: list[ConstraintViolation] = []
    candidate_folded = candidate.casefold()
    for constraint in constraints:
        if constraint.kind is ConstraintKind.REQUIRED:
            passed = constraint.value.casefold() in candidate_folded
        elif constraint.kind is ConstraintKind.FORBIDDEN:
            passed = constraint.value.casefold() not in candidate_folded
        elif constraint.kind is ConstraintKind.PATTERN_REQUIRED:
            passed = re.search(constraint.value, candidate, flags=re.IGNORECASE) is not None
        else:
            passed = re.search(constraint.value, candidate, flags=re.IGNORECASE) is None
        if not passed:
            gate *= constraint.factor
            violations.append(
                ConstraintViolation(
                    name=constraint.name,
                    factor=constraint.factor,
                    reason=f"{constraint.kind.value} constraint failed",
                )
            )
    return gate, tuple(violations)
