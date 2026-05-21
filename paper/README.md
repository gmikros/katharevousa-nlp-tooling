# Paper — Katharevousa NLP

LaTeX source for the manuscript that accompanies the `kathnlp` project.

## Files

- `main.tex` — full manuscript (single-file build)
- `references.bib` — bibliography
- `figures/` — figure assets included from `main.tex`

## Build

Locally (any TeX Live distribution):

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Overleaf

Import the GitHub repository (`gmikros/katharevousa-nlp-tooling`) into Overleaf and set `paper/main.tex` as the main document. After updates are pushed to GitHub, refresh the Overleaf copy via **Menu → Git → Pull GitHub changes**.

See the top-level [`README.md`](../README.md) for project context, results, and citation.
