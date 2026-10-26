"""Command-line interface for one-off SAIS scoring."""

from __future__ import annotations

import argparse
import json

from .core import SAISScorer
from .encoders import HashingEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a candidate against a reference with SAIS.")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--fact", action="append", default=[], help="Required fact; repeat as needed.")
    parser.add_argument("--encoder", choices=("transformer", "hashing"), default="transformer")
    arguments = parser.parse_args()
    encoder = HashingEncoder() if arguments.encoder == "hashing" else None
    result = SAISScorer(encoder=encoder).score(
        arguments.candidate,
        arguments.reference,
        required_facts=arguments.fact or None,
    )
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
