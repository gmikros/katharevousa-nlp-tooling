# Katharevousa NLP Tooling







UD-based semi-manual pipeline for Katharevousa morphology and syntax.







## Pipeline Steps







1. Export source files:



   - `python scripts/export_sources.py --docx-path "<path to Appendix_C.docx>" --excel-path "<path to Excel>.xlsx" --output-dir data`



2. Reconstruct wrapped/hyphenated OCR text before annotation:



   - `python scripts/reconstruct_answer_files.py --csv-paths data/exports/sheets/1976.csv data/exports/sheets/1977.csv --answer-column-name "Answer Files" --output-dir data/interim/reconstructed_sheets`



3. Build pilot dataset with majority vote from reconstructed text:



   - `python scripts/build_pilot_dataset.py --csv-paths data/interim/reconstructed_sheets/1976_reconstructed.csv data/interim/reconstructed_sheets/1977_reconstructed.csv --answer-column-name "Answer Files Reconstructed" --sample-size 1000 --output-dir data/processed/pilot`



4. Resolve disagreements to provisional gold:



   - `python scripts/adjudicate_disagreements.py --pilot-dir data/processed/pilot --strategy openai_first`



5. Train and evaluate models:



   - `python scripts/train_and_evaluate.py`



   - Transformer parser (XLM-R baseline):



     `python scripts/train_transformer_parser.py --gold-path data/processed/final_gold/gold_final.conllu --encoder-name xlm-roberta-base --epochs 6 --batch-size 8`



6. Tune and expand active-learning batches:



   - `python scripts/tune_and_expand.py --csv-paths data/exports/sheets/1976.csv data/exports/sheets/1977.csv --pilot-dir data/processed/pilot --output-dir data/processed/scaleup`



7. Run an interactive copilot menu for the full experiment:



   - `python scripts/copilot_session.py`



   - Includes API key prompt, staged execution, and disagreement review loop.



8. Resume GPT batch generation after interruption/reboot:



   - `python scripts/resume_gpt_gold_batches.py --max-batches 1`



   - Re-running this command is safe; it auto-detects the next unfinished offset.



   - For longer unattended runs: `python scripts/resume_gpt_gold_batches.py --max-batches 10`



   - Resume metadata is written to `data/processed/gold_gpt5_batches/resume_state.json`.







## LLM Configuration







Set API keys to use real model voting:







- `OPENAI_API_KEY` (uses `OPENAI_MODEL`, default `gpt-5.5`)



- `ANTHROPIC_API_KEY` (uses `ANTHROPIC_MODEL`, default `claude-opus-4-7-thinking`)



- `GEMINI_API_KEY` (uses `GEMINI_MODEL`, default `gemini-2.5-pro`)







If keys are not set, the pipeline runs in deterministic heuristic fallback mode for testing.







## Main Artifacts







- `data/exports/export_summary.json`



- `data/interim/reconstructed_sheets/reconstruction_summary.json`



- `data/interim/reconstructed_sheets/reconstruction_audit.json`



- `data/processed/pilot/auto_accepted.conllu`



- `data/processed/pilot/disagreements.json`



- `data/processed/pilot/gold_provisional.conllu`



- `data/processed/pilot/human_decisions.json`



- `reports/pilot_eval_report.json`



- `data/processed/scaleup/tuning_report.json`



