# Third-party notices and distribution checklist

SAIS source code is proprietary under `EULA.md`. The following components are
not relicensed by SAIS. This manifest is an operational checklist, not legal
advice. Verify the exact version, license text, notices, and distribution
obligations before every release.

## Recommended local deployment

The safest commercial delivery is a SAIS-only wheel. Customers separately
install optional Python packages, Ollama, and their chosen embedding model.
This avoids bundling third-party binaries or model weights with SAIS.

## Optional Python dependencies

The declared optional dependencies are generally distributed under permissive
licenses compatible with proprietary applications:

| Component | Typical upstream license | Commercial delivery action |
|---|---|---|
| `sentence-transformers` | Apache-2.0 | Retain its license/notice if bundled. |
| `spacy` | MIT | Retain its license/notice if bundled. |
| `bert-score` | MIT | Retain its license/notice if bundled. |
| `datasets` | Apache-2.0 | Retain its license/notice if bundled. |
| `rouge-score` | Apache-2.0 | Retain its license/notice if bundled. |
| `sacrebleu` | Apache-2.0 | Retain its license/notice if bundled. |
| `scipy` | BSD-style | Retain the SciPy and bundled-binary notices if bundled. |
| `torch` | BSD-style | Retain its license/notice if bundled. |
| `transformers` | Apache-2.0 | Retain its license/notice if bundled. |
| `pytest` | MIT | Development dependency, do not ship unless needed. |

Transitive dependencies have their own licenses. Generate a software bill of
materials and review it for the exact release environment before distribution.

## Ollama and model weights

Ollama and `qwen3-embedding:0.6b` are obtained separately by the customer and
are not included in the SAIS wheel. Their terms control use and redistribution.
Before referencing the model in sales material or bundling it, capture the
official model card and license for the exact model tag and quantization.

## Benchmark data

WMT22 MQM data is for evaluation only. Do not package, redistribute, or use it
in a commercial customer delivery without verifying its dataset terms.
