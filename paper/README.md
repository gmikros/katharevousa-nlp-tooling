# Paper Folder (Journal-style, Overleaf-ready)

This folder is structured for direct Overleaf import from GitHub with modular section files.

## Structure

- `main.tex` - master manuscript file
- `references.bib` - bibliography
- `sections/` - one file per major section
- `tables/` - reusable table blocks included by sections
- `figures/` - image assets (`.pdf`, `.png`, `.eps`)

## Local Editing and GitHub Sync

- **Local paper folder:** `C:\Users\USER01\Dropbox\Workplace\D\George\PAPERS\Katharevousa NLP library\paper`
- **GitHub repository:** [github.com/gmikros/katharevousa-nlp-tooling](https://github.com/gmikros/katharevousa-nlp-tooling)

Edit LaTeX files locally, then push from the repository root:

```powershell
cd "C:\Users\USER01\Dropbox\Workplace\D\George\PAPERS\Katharevousa NLP library"
git add paper/
git commit -m "Update paper section"
git push origin main
```

## Overleaf Import

1. Overleaf → **New Project** → **Import from GitHub**
2. Select `gmikros/katharevousa-nlp-tooling`
3. Set `paper/main.tex` as the main document if auto-detection fails
4. After local pushes, pull in Overleaf via **Menu → Git → Pull GitHub changes**

## Editing Conventions

- Put narrative text in `sections/*.tex`.
- Put large tables in `tables/*.tex` and include with `\input{paper/tables/...}`.
- Keep figures in `figures/` and reference from section files.
- Use citation keys from `references.bib` only; add missing entries before submission.
