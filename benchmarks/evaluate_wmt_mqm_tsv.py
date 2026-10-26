"""Benchmark SAIS against the official WMT22 expert MQM TSV release.

The Google WMT MQM repository includes candidate translations, references, and
average segment scores. The released scores are negative error penalties, so
values closer to zero are better and can be correlated directly with SAIS.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from scipy.stats import pearsonr, spearmanr

from sais_score import HashingEncoder, OllamaEncoder, SAISScorer, SentenceTransformerEncoder


def _correlation(scores: list[float], human: list[float]) -> dict[str, float | int]:
    if len(scores) < 3:
        raise ValueError("At least three observations are required for correlation.")
    return {
        "observations": len(scores),
        "pearson": float(pearsonr(scores, human).statistic),
        "spearman": float(spearmanr(scores, human).statistic),
    }


def _rows_by_system(path: Path, per_system: int | None) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        # WMT text may contain literal double quotes, not CSV-escaped fields.
        # Treat each physical tab-delimited line as a separate record.
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row["score"] in {"", "None"} or not row.get("hyp") or not row.get("ref"):
                continue
            if per_system is None or len(grouped[row["sys"]]) < per_system:
                grouped[row["sys"]].append(row)
    if not grouped:
        raise ValueError(f"No TSV rows were found in {path}.")
    return dict(grouped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SAIS against WMT expert MQM annotations.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument(
        "--per-system",
        type=int,
        default=100,
        help="Use this many first segments per system. Use 0 for all segments.",
    )
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument(
        "--encoder",
        choices=("transformer", "ollama", "hashing"),
        default="transformer",
        help=(
            "Use transformer or ollama for a reportable SAIS evaluation. "
            "Hashing is an offline pipeline smoke test only and is not a "
            "semantic model."
        ),
    )
    parser.add_argument("--ollama-model", default="qwen3-embedding:0.6b")
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, default=Path("wmt22_mqm_sais.json"))
    args = parser.parse_args()
    per_system = None if args.per_system == 0 else args.per_system
    if per_system is not None and per_system < 3:
        raise ValueError("--per-system must be at least 3, or 0 for all.")

    grouped = _rows_by_system(args.dataset, per_system)
    if args.encoder == "hashing":
        encoder = HashingEncoder()
    elif args.encoder == "ollama":
        encoder = OllamaEncoder(args.ollama_model, args.ollama_url)
    else:
        encoder = SentenceTransformerEncoder(args.model)
    scorer = SAISScorer(encoder=encoder)
    system_scores: dict[str, dict[str, float | int]] = {}
    metric_values: list[float] = []
    human_values: list[float] = []
    for system, rows in sorted(grouped.items()):
        sais_values = [scorer.score(row["hyp"], row["ref"]).score for row in rows]
        # MQM uses negative error penalties, so higher raw values are better.
        human_quality = [float(row["score"]) for row in rows]
        system_scores[system] = {
            "segments": len(rows),
            "sais_mean": mean(sais_values),
            "human_quality_mean": mean(human_quality),
        }
        metric_values.extend(sais_values)
        human_values.extend(human_quality)

    system_metric = [float(value["sais_mean"]) for value in system_scores.values()]
    system_human = [float(value["human_quality_mean"]) for value in system_scores.values()]
    report = {
        "dataset": str(args.dataset),
        "dataset_description": "WMT22 expert MQM average segment scores",
        "metric": "SAIS",
        "encoder": args.encoder,
        "embedding_model": (
            "HashingEncoder (non-semantic smoke test)"
            if args.encoder == "hashing"
            else args.ollama_model if args.encoder == "ollama" else args.model
        ),
        "reportable_as_empirical_sais": args.encoder != "hashing",
        "evaluation_scope": "complete_dataset"
        if per_system is None
        else "deterministic_preliminary_subset",
        "suitable_for_final_benchmark_claim": args.encoder != "hashing" and per_system is None,
        "human_score_transform": "raw MQM score; negative error penalties, higher is better",
        "systems": len(system_scores),
        "segments_per_system_limit": per_system or "all",
        "segment_level_correlation": _correlation(metric_values, human_values),
        "system_level_correlation": _correlation(system_metric, system_human),
        "system_scores": system_scores,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
