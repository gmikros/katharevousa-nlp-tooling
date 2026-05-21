# Maintainer Notes

Operational notes for the project maintainer. Most users do not need this file; see the top-level [`README.md`](../README.md) for installation, quick start, and citation.

## End-to-end corpus reconstruction

Rebuilding the frozen reference set from the raw archival exports:

1. **Export the source documents** (DOCX + XLSX) into per-sheet CSVs:

   ```bash
   python scripts/export_sources.py \
     --docx-path "<appendix_docx>" \
     --excel-path "<source_xlsx>" \
     --output-dir data
   ```

2. **Reconstruct OCR-derived answer text** from the spreadsheet columns:

   ```bash
   python scripts/reconstruct_answer_files.py \
     --csv-paths data/exports/sheets/1976.csv data/exports/sheets/1977.csv \
     --answer-column-name "Answer Files" \
     --output-dir data/interim/reconstructed_sheets
   ```

3. **Freeze the final gold snapshot** (deterministic merge of batch and retry outputs):

   ```bash
   python scripts/freeze_final_gold_snapshot.py
   ```

4. **Train and evaluate baselines and the release candidate**:

   ```bash
   python scripts/train_and_evaluate.py
   python scripts/train_transformer_parser.py \
     --gold-path data/processed/final_gold/gold_final.conllu \
     --encoder-name xlm-roberta-base \
     --epochs 3 --batch-size 4
   ```

5. **Evaluate external library baselines** under the same split:

   ```bash
   python scripts/evaluate_external_baselines.py
   ```

## Released benchmark reports

The release-candidate scores in the README come from the following JSON files under `reports/`:

- `transformer_parser_v1_opt_report.json` — XLM-R release candidate
- `transformer_parser_mbert_v2_report.json` — mBERT comparison
- `stanza_custom_v1_report.json` — custom-trained Stanza
- `stanza_baseline_report.json` — pretrained Stanza (Greek and Ancient Greek PROIEL)
- `external_baselines_report.json` — spaCy Greek + Stanza external baselines
- `final_gold_eval_report.json` — feature-based baseline
- `transformer_parser_v1_opt_split.json` — seed-42 fixed split used for all scores

## Working copy and GitHub sync

The maintainer working copy lives outside this repository, under a synced Dropbox folder:

```
C:\Users\USER01\Dropbox\Workplace\D\George\PAPERS\Katharevousa NLP library
```

Use the helper script for the routine push/pull cycle:

```powershell
.\scripts\sync_github.ps1 status
.\scripts\sync_github.ps1 pull
.\scripts\sync_github.ps1 push -Message "Describe your change"
```

Equivalent manual commands:

```powershell
cd "C:\Users\USER01\Dropbox\Workplace\D\George\PAPERS\Katharevousa NLP library"
git pull origin main
git add -A
git commit -m "Describe your change"
git push origin main
```

## Overleaf

Overleaf imports the manuscript from GitHub (`gmikros/katharevousa-nlp-tooling`), not from Dropbox. After pushing from this folder, refresh in Overleaf via **Menu → Git → Pull GitHub changes**. The Overleaf main document is `paper/main.tex`.
