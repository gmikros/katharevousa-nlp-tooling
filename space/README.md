---
title: kathnlp - Katharevousa Greek Parser
emoji: 🏛️
colorFrom: indigo
colorTo: yellow
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Universal-Dependencies parser for Katharevousa Greek
tags:
  - katharevousa
  - greek
  - dependency-parsing
  - universal-dependencies
  - historical-nlp
models:
  - gmikros/kathnlp-xlmr
datasets:
  - gmikros/kathnlp-treebank
arxiv: 2605.22978
---

# kathnlp · Katharevousa Greek dependency parser

Interactive demo of [`gmikros/kathnlp-xlmr`](https://huggingface.co/gmikros/kathnlp-xlmr), a fine-tuned XLM-RoBERTa parser for **Katharevousa Greek**, the archaizing official register used in 20th-century Greek law, administration, and parliamentary discourse.

Paste a sentence (or pick an example) and the model returns UPOS tags, dependency arcs, and a CoNLL-U-style table.

- **Paper:** [arXiv:2605.22978](https://arxiv.org/abs/2605.22978)
- **Code:** <https://github.com/gmikros/katharevousa-nlp-tooling>
- **Treebank:** [`gmikros/kathnlp-treebank`](https://huggingface.co/datasets/gmikros/kathnlp-treebank)

## Notes

- Running on the free CPU tier; the first request after a cold start may take 30–60 seconds while the model is downloaded.
- Sentences are whitespace-tokenized before parsing.
- Sequence length capped at 256 subword tokens (~50–80 Greek words).
