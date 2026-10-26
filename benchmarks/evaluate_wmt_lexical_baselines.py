"""Evaluate BLEU and ROUGE-L against WMT MQM human scores.

Uses the same WMT22 expert-MQM TSV parsing and score orientation as
``evaluate_wmt_mqm_tsv.py``: higher raw MQM values (closer to zero) are better.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import sacrebleu
from rouge_score.rouge_scorer import RougeScorer
from scipy.stats import pearsonr, spearmanr


def _correlation(scores: list[float], human: list[float]) -> dict[str, float | int]:
    return {
        "observations": len(scores),
        "pearson": float(pearsonr(scores, human).statistic),
        "spearman": float(spearmanr(scores, human).statistic),
    }


def _rows_by_system(path: Path) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE):
            if row["score"] in {"", "None"} or not row.get("hyp") or not row.get("ref"):
                continue
            grouped[row["sys"]].append(row)
    return dict(grouped)


def _report_metric(
    grouped: dict[str, list[dict[str, str]]],
    name: str,
    score_fn: object,
) -> dict[str, object]:
    system_scores: dict[str, dict[str, float | int]] = {}
    metric_values: list[float] = []
    human_values: list[float] = []
    for system, rows in sorted(grouped.items()):
        values = [score_fn(row["hyp"], row["ref"]) for row in rows]  # type: ignore[operator]
        human = [float(row["score"]) for row in rows]
        system_scores[system] = {
            "segments": len(rows),
            "metric_mean": mean(values),
            "human_quality_mean": mean(human),
        }
        metric_values.extend(values)
        human_values.extend(human)
    return {
        "segment_level_correlation": _correlation(metric_values, human_values),
        "system_level_correlation": _correlation(
            [float(item["metric_mean"]) for item in system_scores.values()],
            [float(item["human_quality_mean"]) for item in system_scores.values()],
        ),
        "system_scores": system_scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate lexical metrics against WMT expert MQM.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("wmt22_mqm_lexical.json"))
    args = parser.parse_args()
    grouped = _rows_by_system(args.dataset)
    rouge = RougeScorer(["rougeL"], use_stemmer=False)
    report = {
        "dataset": str(args.dataset),
        "dataset_description": "WMT22 expert MQM average segment scores",
        "evaluation_scope": "complete_dataset",
        "human_score_transform": "raw MQM score; negative error penalties, higher is better",
        "systems": len(grouped),
        "metrics": {
            "bleu": _report_metric(
                grouped,
                "BLEU",
                lambda hypothesis, reference: sacrebleu.sentence_bleu(
                    hypothesis, [reference]
                ).score
                / 100,
            ),
            "rougeL": _report_metric(
                grouped,
                "ROUGE-L",
                lambda hypothesis, reference: rouge.score(reference, hypothesis)["rougeL"].fmeasure,
            ),
        },
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
