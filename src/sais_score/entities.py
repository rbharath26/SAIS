"""Fact extraction, canonicalization, and one-to-one matching."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

_DATE_FORMATS = ("%Y-%m-%d", "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y")
_DATE_PATTERN = re.compile(
    r"\b(?:\d{4}-\d{2}-\d{2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|"
    r"Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?) \d{1,2}, \d{4})\b",
    re.IGNORECASE,
)
_NUMBER_PATTERN = re.compile(r"\b\d+(?:[,.]\d+)?%?\b")
_PROPER_NOUN_PATTERN = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def canonicalize_fact(value: str, aliases: dict[str, str] | None = None) -> str:
    """Normalize common date, number, case, and whitespace variants."""
    normalized = " ".join(value.strip().split()).casefold()
    if aliases and normalized in aliases:
        normalized = aliases[normalized].casefold()
    for date_format in _DATE_FORMATS:
        try:
            return "date:" + datetime.strptime(normalized.title(), date_format).date().isoformat()
        except ValueError:
            pass
    numeric = normalized.replace(",", "")
    if re.fullmatch(r"\d+(?:\.\d+)?%?", numeric):
        suffix = "%" if numeric.endswith("%") else ""
        numeric = numeric.removesuffix("%")
        number = float(numeric)
        rendered = str(int(number)) if number.is_integer() else f"{number:g}"
        return f"number:{rendered}{suffix}"
    return "text:" + normalized


def extract_facts(text: str, use_spacy: bool = False) -> list[str]:
    """Extract dates, numbers, proper names, and optionally spaCy entities.

    Automatic extraction is a fallback. For high-stakes evaluation, callers
    should pass curated ``required_facts`` and explicit constraints.
    """
    date_matches = list(_DATE_PATTERN.finditer(text))
    facts = [match.group(0) for match in date_matches]
    for number_match in _NUMBER_PATTERN.finditer(text):
        is_part_of_date = any(
            date_match.start() <= number_match.start()
            and number_match.end() <= date_match.end()
            for date_match in date_matches
        )
        if not is_part_of_date:
            facts.append(number_match.group(0))
    facts.extend(_PROPER_NOUN_PATTERN.findall(text))
    if use_spacy:
        try:
            import spacy

            nlp = spacy.load("en_core_web_sm")
            facts.extend(entity.text for entity in nlp(text).ents)
        except (ImportError, OSError) as error:
            raise RuntimeError(
                "spaCy extraction requires `pip install sais-score[ner]` and "
                "`python -m spacy download en_core_web_sm`."
            ) from error
    return _unique_preserving_order(facts)


def _unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        key = canonicalize_fact(value)
        if key not in seen:
            seen.add(key)
            output.append(value)
    return output


@dataclass(frozen=True)
class FactMatch:
    matched: tuple[str, ...]
    missing: tuple[str, ...]
    unmatched_candidate: tuple[str, ...]


def match_facts(
    reference_facts: Iterable[str],
    candidate_facts: Iterable[str],
    aliases: dict[str, str] | None = None,
) -> FactMatch:
    """One-to-one exact canonical matching, preventing duplicate credit."""
    reference = list(_unique_preserving_order(reference_facts))
    candidate = list(_unique_preserving_order(candidate_facts))
    by_canonical: dict[str, list[int]] = {}
    for index, value in enumerate(candidate):
        by_canonical.setdefault(canonicalize_fact(value, aliases), []).append(index)

    used: set[int] = set()
    matched: list[str] = []
    missing: list[str] = []
    for value in reference:
        key = canonicalize_fact(value, aliases)
        match_index = next((item for item in by_canonical.get(key, []) if item not in used), None)
        if match_index is None:
            missing.append(value)
        else:
            used.add(match_index)
            matched.append(value)
    unmatched = [value for index, value in enumerate(candidate) if index not in used]
    return FactMatch(tuple(matched), tuple(missing), tuple(unmatched))
