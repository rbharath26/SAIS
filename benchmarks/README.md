# Benchmark protocol

The benchmark harness intentionally does not download or redistribute evaluation
sets. Dataset licenses and human annotations must be reviewed before use.

1. Obtain a human-scored reference-based dataset, for example an eligible WMT
   Metrics Shared Task release.
2. Convert each scored segment to JSONL:

   ```json
   {"candidate":"system output","reference":"reference translation","human_score":78.0}
   ```

3. Install extras and run:

   ```bash
   py -m pip install -e .[embeddings,benchmarks]
   py benchmarks/evaluate_jsonl.py data.jsonl --include-baselines --output results.json
   ```

4. Record dataset version, language pair, preprocessing, model version,
   hyperparameters, segment count, and Pearson/Spearman results.

Do not add numerical claims to the preprint until this command has produced
the corresponding stored `results.json`.

## WMT system-level protocol

The official Google `mt-metrics-eval` toolkit provides WMT data and compatible
human scores. Its database is large, so download it only after reviewing its
license and storage requirements:

```bash
py -m pip install mt-metrics-eval
py -m mt_metrics_eval.mtme --download
py -m pip install -e .[embeddings]
py benchmarks/evaluate_wmt.py --test-set wmt22 --language-pair en-de \
  --output wmt22-en-de-system.json
```

For a non-reportable smoke test, pass `--max-segments 20`. The script records
the exact test set, language pair, human-score field, embedding model, systems,
and segment count. It uses WMT's stored BLEU/BERTScore variants only when an
exact reference-compatible variant is present. Use `evaluate_jsonl.py` with
the optional `benchmarks` extra to calculate ROUGE-L and BERTScore directly.

## WMT22 expert MQM release

The smaller [Google WMT MQM annotation release](https://github.com/google/wmt-mqm-human-evaluation)
contains WMT22 English-to-German hypotheses, references, and expert MQM
annotations in one TSV. It avoids downloading the full MTME archive:

```bash
py -m pip install -e .[embeddings]
py benchmarks/evaluate_wmt_mqm_tsv.py \
  vendor/wmt-mqm-human-evaluation/generalMT2022/ende/\
mqm_generalMT2022_ende.avg_seg_scores.tsv \
  --per-system 100 --output wmt22-mqm-preliminary.json
```

`--per-system 0` runs all rows. The default 100-row mode is explicitly
a preliminary, deterministic subset and should not be presented as the final
benchmark result.

If a transformer model cannot be downloaded, add `--encoder hashing` to test
the complete scoring and reporting pipeline offline. Its JSON result has
`reportable_as_empirical_sais: false`; it is validation evidence, not an SAIS
benchmark result.

## Local Ollama embedding evaluation

With `qwen3-embedding:0.6b` pulled locally, run the same benchmark without
Hugging Face:

```bash
py benchmarks/evaluate_wmt_mqm_tsv.py \
  vendor/wmt-mqm-human-evaluation/generalMT2022/ende/\
mqm_generalMT2022_ende.avg_seg_scores.tsv \
  --per-system 100 --encoder ollama \
  --ollama-model qwen3-embedding:0.6b \
  --output results/wmt22-mqm-qwen3-preliminary.json
```

Use `--per-system 0` for the complete dataset. Record the Ollama and model
versions with every reported correlation result.

## Lexical baselines

Run BLEU and ROUGE-L over the complete identical WMT22 rows:

```bash
py benchmarks/evaluate_wmt_lexical_baselines.py \
  vendor/wmt-mqm-human-evaluation/generalMT2022/ende/\
mqm_generalMT2022_ende.avg_seg_scores.tsv \
  --output results/wmt22-mqm-lexical.json
```

BERTScore requires a separately available contextual model. Do not substitute
the hashing smoke encoder for BERTScore or report it as a semantic baseline.

After both complete-dataset reports are available, consolidate them:

```bash
py benchmarks/consolidate_validation.py \
  results/wmt22-mqm-qwen3-full.json results/wmt22-mqm-lexical.json \
  --output results/wmt22-mqm-validation.md
```
