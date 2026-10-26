# SAIS validation, WMT22 expert MQM

- **SAIS embedding model:** `qwen3-embedding:0.6b`
- **Dataset:** `vendor\wmt-mqm-human-evaluation\generalMT2022\ende\mqm_generalMT2022_ende.avg_seg_scores.tsv`
- **Human score:** raw MQM negative error penalty, where larger values are better.
- **Scope:** complete valid WMT22 English-to-German expert-MQM rows.

| Metric | Segments | Segment Pearson | Segment Spearman | System Pearson | System Spearman |
|---|---:|---:|---:|---:|---:|
| SAIS | 19725 | 0.075 | 0.189 | 0.727 | 0.539 |
| BLEU | 19725 | 0.199 | 0.228 | 0.632 | 0.457 |
| ROUGE-L | 19725 | 0.227 | 0.258 | 0.712 | 0.500 |

## Limitation

BERTScore is omitted because this environment cannot establish TLS with Hugging Face to obtain its contextual model. These results must not be presented as a complete three-baseline comparison until BERTScore is run under the same protocol.
