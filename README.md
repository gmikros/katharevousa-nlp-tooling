# Katharevousa NLP Tooling

Reproducible tooling for morphological and dependency parsing experiments on Katharevousa Greek parliamentary text.

## Overview

This repository provides the complete research pipeline from OCR-derived source reconstruction to gold-data freezing, model training, and benchmark comparison.

Core components include:

- text reconstruction for historical OCR artifacts;
- schema-constrained annotation workflows;
- deterministic CoNLL-U snapshot generation;
- fixed-split evaluation across model families;
- paper and release assets for scientific dissemination.

## Repository Layout

- `configs/` - annotation schema and run configuration
- `scripts/` - data processing, training, and evaluation entrypoints
- `src/kathnlp/` - library code for pipelines, training, and metrics
- `reports/` - experiment outputs and comparative benchmarks
- `paper/` - journal-style LaTeX manuscript

## Reproducible Workflow

1. Export source files:
   - `python scripts/export_sources.py --docx-path "<appendix_docx>" --excel-path "<source_xlsx>" --output-dir data`
2. Reconstruct OCR text:
   - `python scripts/reconstruct_answer_files.py --csv-paths data/exports/sheets/1976.csv data/exports/sheets/1977.csv --answer-column-name "Answer Files" --output-dir data/interim/reconstructed_sheets`
3. Freeze final gold snapshot:
   - `python scripts/freeze_final_gold_snapshot.py`
4. Train and evaluate parser baselines:
   - `python scripts/train_and_evaluate.py`
   - `python scripts/train_transformer_parser.py --gold-path data/processed/final_gold/gold_final.conllu --encoder-name xlm-roberta-base --epochs 3 --batch-size 4`
5. Evaluate external library baselines:
   - `python scripts/evaluate_external_baselines.py`

## Key Reports

- `reports/transformer_parser_v1_opt_report.json`
- `reports/transformer_parser_mbert_v2_report.json`
- `reports/stanza_custom_v1_report.json`
- `reports/stanza_baseline_report.json`
- `reports/external_baselines_report.json`

## Paper

The manuscript source is maintained in `paper/` and is structured for direct Overleaf import from GitHub.
