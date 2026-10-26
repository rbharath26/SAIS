"""Evaluate SAIS against WMT human scores through mt-metrics-eval.

The default is a system-level WMT22 en-de evaluation. Use a multilingual
SentenceTransformer model for translation pairs. The command deliberately
requires an explicit --max-segments value when prototyping, and reports all
settings in its JSON output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from sais_score import SAISScorer, SentenceTransformerEncoder


def _correlations(metric: dict[str, float], human: dict[str, float]) -> dict[str, float]:
    from scipy.stats import pearsonr, spearmanr

    systems = sorted(set(metric) & set(human))
    if len(systems) < 3:
        raise ValueError("At least three systems with both metric and human scores are required.")
    metric_values = [metric[system] for system in systems]
    human_values = [human[system] for system in systems]
    return {
        "systems": len(systems),
        "pearson": float(pearsonr(metric_values, human_values).statistic),
        "spearman": float(spearmanr(metric_values, human_values).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Correlate SAIS with WMT human scores.")
    parser.add_argument("--test-set", default="wmt22")
    parser.add_argument("--language-pair", default="en-de")
    parser.add_argument(
        "--model",
        default="paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence-transformers model. The default supports multiple languages.",
    )
    parser.add_argument(
        "--max-segments",
        type=int,
        default=None,
        help="Optional deterministic prefix size per system for a smoke test.",
    )
    parser.add_argument("--output", type=Path, default=Path("wmt_sais_results.json"))
    args = parser.parse_args()

    try:
        from mt_metrics_eval import data
    except ImportError as error:
        raise RuntimeError("Install mt-metrics-eval and its data first.") from error

    evaluation = data.EvalSet(args.test_set, args.language_pair, read_stored_metric_scores=True)
    reference = evaluation.all_refs[evaluation.std_ref]
    system_names = sorted(evaluation.sys_names - evaluation.human_sys_names - evaluation.outlier_sys_names)
    segment_count = min(len(reference), args.max_segments or len(reference))
    reference = reference[:segment_count]

    scorer = SAISScorer(encoder=SentenceTransformerEncoder(args.model))
    sais_by_system: dict[str, float] = {}
    for system in system_names:
        outputs = evaluation.sys_outputs[system][:segment_count]
        sais_by_system[system] = mean(
            scorer.score(candidate, target).score
            for candidate, target in zip(outputs, reference, strict=True)
        )

    human_name = evaluation.StdHumanScoreName("sys")
    human_by_system = {
        system: scores[0]
        for system, scores in evaluation.Scores("sys", human_name).items()
        if system in system_names and scores[0] is not None
    }
    report: dict[str, object] = {
        "dataset": args.test_set,
        "language_pair": args.language_pair,
        "level": "system",
        "human_score": human_name,
        "systems_considered": len(system_names),
        "segments_per_system": segment_count,
        "embedding_model": args.model,
        "sais": {
            "system_scores": sais_by_system,
            "correlation": _correlations(sais_by_system, human_by_system),
        },
    }

    # Use stored scores for the directly comparable WMT baselines when present.
    for base_name in ("BLEU", "BERTScore"):
        metric_name = next(
            (
                name
                for name in evaluation.metric_names
                if evaluation.BaseMetric(name) == base_name
                and evaluation.ReferencesUsed(name) == {evaluation.std_ref}
            ),
            None,
        )
        if metric_name:
            stored = {
                system: scores[0]
                for system, scores in evaluation.Scores("sys", metric_name).items()
                if system in system_names and scores[0] is not None
            }
            report[base_name.lower()] = {
                "metric_name": metric_name,
                "correlation": _correlations(stored, human_by_system),
            }

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
