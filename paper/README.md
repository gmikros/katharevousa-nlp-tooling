# Paper Folder (Journal-style, Overleaf-ready)

This folder is structured for direct Overleaf import from GitHub with modular section files.

## Structure

- `main.tex` - master manuscript file
- `references.bib` - bibliography
- `sections/` - one file per major section
- `tables/` - reusable table blocks included by sections
- `figures/` - image assets (`.pdf`, `.png`, `.eps`)

## Overleaf Import

1. Overleaf -> `New Project` -> `Import from GitHub`
2. Select this repository (`gmikros/katharevousa-nlp-tooling`)
3. In Overleaf, set `paper/main.tex` as the main document if auto-detection fails

## Editing Conventions

- Put narrative text in `sections/*.tex`.
- Put large tables in `tables/*.tex` and include with `\input{tables/...}`.
- Keep figures in `figures/` and reference from section files.
- Use citation keys from `references.bib` only; add missing entries before submission.
