"""Create a concise validation report from complete SAIS and lexical results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(name: str, result: dict[str, object]) -> str:
    segment = result["segment_level_correlation"]  # type: ignore[index]
    system = result["system_level_correlation"]  # type: ignore[index]
    return (
        f"| {name} | {segment['observations']} | {segment['pearson']:.3f} | "
        f"{segment['spearman']:.3f} | {system['pearson']:.3f} | {system['spearman']:.3f} |"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sais", type=Path)
    parser.add_argument("lexical", type=Path)
    parser.add_argument("--output", type=Path, default=Path("validation_report.md"))
    args = parser.parse_args()
    sais = _read(args.sais)
    lexical = _read(args.lexical)
    if sais.get("evaluation_scope") != "complete_dataset":
        raise ValueError("SAIS input must be a complete-dataset evaluation, not a preliminary subset.")
    if not sais.get("reportable_as_empirical_sais"):
        raise ValueError("SAIS input uses a non-semantic encoder and is not reportable.")

    metrics = lexical["metrics"]  # type: ignore[index]
    content = "\n".join(
        [
            "# SAIS validation, WMT22 expert MQM",
            "",
            f"- **SAIS embedding model:** `{sais['embedding_model']}`",
            f"- **Dataset:** `{sais['dataset']}`",
            "- **Human score:** raw MQM negative error penalty, where larger values are better.",
            "- **Scope:** complete valid WMT22 English-to-German expert-MQM rows.",
            "",
            "| Metric | Segments | Segment Pearson | Segment Spearman | System Pearson | System Spearman |",
            "|---|---:|---:|---:|---:|---:|",
            _row("SAIS", sais),
            _row("BLEU", metrics["bleu"]),  # type: ignore[index]
            _row("ROUGE-L", metrics["rougeL"]),  # type: ignore[index]
            "",
            "## Limitation",
            "",
            "BERTScore is omitted because this environment cannot establish TLS with Hugging Face to obtain its contextual model. These results must not be presented as a complete three-baseline comparison until BERTScore is run under the same protocol.",
            "",
        ]
    )
    args.output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
