"""Reproducibly correlate SAIS and baseline metrics with human scores.

Input is JSONL with one object per segment:
{"candidate": "...", "reference": "...", "human_score": 0.0}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from sais_score import SAISScorer


def _load_rows(path: Path) -> list[dict[str, object]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"candidate", "reference", "human_score"}
    if not rows or any(not required <= row.keys() for row in rows):
        raise ValueError(f"{path} must contain JSONL objects with {sorted(required)}.")
    return rows


def _correlations(scores: list[float], human: list[float]) -> dict[str, float]:
    try:
        from scipy.stats import pearsonr, spearmanr
    except ImportError as error:
        raise RuntimeError("Install benchmark dependencies: pip install -e .[benchmarks]") from error
    return {
        "pearson": float(pearsonr(scores, human).statistic),
        "spearman": float(spearmanr(scores, human).statistic),
    }


def _baselines(candidates: list[str], references: list[str]) -> dict[str, list[float]]:
    try:
        from bert_score import score as bert_score
        from rouge_score.rouge_scorer import RougeScorer
        import sacrebleu
    except ImportError as error:
        raise RuntimeError("Install benchmark dependencies: pip install -e .[benchmarks]") from error
    rouge = RougeScorer(["rougeL"], use_stemmer=True)
    _, _, bert_f1 = bert_score(candidates, references, lang="en", verbose=True)
    return {
        "bleu": [sacrebleu.sentence_bleu(c, [r]).score / 100 for c, r in zip(candidates, references)],
        "rougeL": [rouge.score(r, c)["rougeL"].fmeasure for c, r in zip(candidates, references)],
        "bertscore": bert_f1.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="JSONL with candidate, reference, human_score.")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args()
    rows = _load_rows(args.dataset)
    candidates = [str(row["candidate"]) for row in rows]
    references = [str(row["reference"]) for row in rows]
    human = [float(row["human_score"]) for row in rows]
    scorer = SAISScorer()
    metrics: dict[str, list[float]] = {
        "sais": [scorer.score(candidate, reference).score for candidate, reference in zip(candidates, references)]
    }
    if args.include_baselines:
        metrics.update(_baselines(candidates, references))
    report = {
        "dataset": str(args.dataset),
        "segments": len(rows),
        "human_score_mean": mean(human),
        "correlations": {name: _correlations(values, human) for name, values in metrics.items()},
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
