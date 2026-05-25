"""Stage the XLM-R kathnlp release candidate for the Hugging Face Hub.

Produces a self-contained `model_release/` directory with:
  - encoder/                 (fine-tuned XLM-R encoder, HF format)
  - tokenizer/               (saved XLM-R tokenizer, HF format)
  - parser_heads.pt          (small custom heads only: UPOS, arc, head, rel)
  - metadata.json            (label maps + training config)
  - README.md                (model card)

The encoder weights live once, in `encoder/model.safetensors`. The
`parser_heads.pt` shipped here contains only the non-encoder parameters,
shrinking the release from ~2.2 GB to ~1.15 GB.

Run from the repository root:

    python scripts/build_model_release.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src" / "kathnlp" / "models" / "transformer_parser_v1_opt"
OUT_DIR = REPO_ROOT / "model_release"

V1_OPT_REPORT = REPO_ROOT / "reports" / "transformer_parser_v1_opt_report.json"


MODEL_CARD = """---
language:
  - el
license: apache-2.0
library_name: kathnlp
base_model: xlm-roberta-base
tags:
  - katharevousa
  - greek
  - historical-greek
  - dependency-parsing
  - universal-dependencies
  - token-classification
  - parsing
  - historical-nlp
  - parliamentary-corpora
  - low-resource
  - xlm-roberta
pipeline_tag: token-classification
datasets:
  - gmikros/kathnlp-treebank
arxiv: 2605.22978
model-index:
  - name: kathnlp-xlmr
    results:
      - task:
          type: token-classification
          name: UPOS tagging
        dataset:
          name: kathnlp Katharevousa treebank (test split, seed 42)
          type: gmikros/kathnlp-treebank
          split: test
        metrics:
          - type: accuracy
            value: 0.8893
            name: UPOS accuracy
      - task:
          type: dependency-parsing
          name: Dependency parsing
        dataset:
          name: kathnlp Katharevousa treebank (test split, seed 42)
          type: gmikros/kathnlp-treebank
          split: test
        metrics:
          - type: f1
            value: 0.7250
            name: DEPREL F1 (weighted)
          - type: accuracy
            value: 0.6098
            name: UAS
          - type: accuracy
            value: 0.5162
            name: LAS
---

# kathnlp-xlmr · Katharevousa Greek dependency parser

`kathnlp-xlmr` is a fine-tuned [XLM-RoBERTa base](https://huggingface.co/xlm-roberta-base) model for **Universal-Dependencies-style morphological tagging and dependency parsing of Katharevousa Greek**, the archaizing official register used in 20th-century Greek law, administration, and parliamentary discourse.

- **Paper:** [arXiv:2605.22978](https://arxiv.org/abs/2605.22978)
- **Code:** <https://github.com/gmikros/katharevousa-nlp-tooling>
- **Dataset:** [`gmikros/kathnlp-treebank`](https://huggingface.co/datasets/gmikros/kathnlp-treebank)
- **Status:** v0.1 research preview — annotations automatically validated, philologist adjudication in progress.

## Results

Evaluated on the fixed 340-sentence test split (seed 42) of the [kathnlp Katharevousa treebank](https://huggingface.co/datasets/gmikros/kathnlp-treebank).

| Metric | Score |
|---|---:|
| UPOS accuracy | **0.8893** |
| DEPREL F1 (weighted) | **0.7250** |
| UAS | **0.6098** |
| LAS | **0.5162** |

These scores outperform every off-the-shelf Greek and Ancient Greek baseline tested in the paper (best external: spaCy Greek with 0.4183 LAS). See the [accompanying paper](https://arxiv.org/abs/2605.22978) for the full benchmark including mBERT, custom-trained Stanza, and the feature-based baseline.

## Usage

The recommended loader is the [`kathnlp`](https://github.com/gmikros/katharevousa-nlp-tooling) package, which downloads the weights and reconstructs the custom parser architecture in one call.

```bash
pip install git+https://github.com/gmikros/katharevousa-nlp-tooling.git
```

```python
from kathnlp.hub import load_from_hub

parser = load_from_hub("gmikros/kathnlp-xlmr")  # add device="cuda" for GPU

text = "Ἡ Κυβέρνησις παρακαλεῖται νά ἀποδεχθῇ τό αἴτημα τοῦ χωρίου."
for tok in parser.parse(text):
    print(tok.id, tok.form, tok.upos, tok.head, tok.deprel)
```

### Raw artifacts

If you would rather load the weights manually:

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="gmikros/kathnlp-xlmr")
# Use:
#   {local_dir}/encoder/                 — fine-tuned XLM-R encoder (HF format)
#   {local_dir}/tokenizer/               — XLM-R tokenizer (HF format)
#   {local_dir}/parser_heads.pt          — custom UPOS, arc, head, and relation heads
#   {local_dir}/metadata.json            — UPOS / DEPREL label maps and training config
```

## Architecture

The parser is a small custom head on top of XLM-R:

- a **UPOS classifier** (linear over the encoder's hidden state),
- two **arc projections** (`dep_arc`, `head_arc`) plus a **root attention** parameter to score every potential head for every word,
- a **relation classifier** that consumes the head and dependent representations to predict the dependency label.

This is *not* a vanilla `AutoModelForTokenClassification`, which is why we ship a small loader (`kathnlp.hub.load_from_hub`) rather than relying on `transformers` auto-discovery.

## Training

- **Base model:** `xlm-roberta-base`
- **Data:** 1,357 training sentences from [`gmikros/kathnlp-treebank`](https://huggingface.co/datasets/gmikros/kathnlp-treebank) (seed 42).
- **Epochs:** 3, batch size 4, learning rate 2 × 10⁻⁵, weight decay 0.01.
- **Max sequence length:** 256 subword tokens.
- **Loss weights:** UPOS 1.0, arc 1.8, relation 1.2.

The exact training script lives at [`scripts/train_transformer_parser.py`](https://github.com/gmikros/katharevousa-nlp-tooling/blob/main/scripts/train_transformer_parser.py) in the repository.

## Limitations

- Trained on 1,357 sentences from 1976–1977 parliamentary questions; transfer to other Katharevousa genres (decrees, newspapers, earlier legal texts) is untested.
- Test split is small (340 sentences / 4,093 tokens).
- Annotations are automatically validated rather than expert-adjudicated; an expert-reviewed v0.2 release is planned.
- Sentence length must fit within 256 XLM-R subword tokens (~50–80 Greek words); longer inputs are truncated.

## License

Released under **Apache 2.0**.

## Citation

```bibtex
@misc{mikrosfitsilis2026kathnlp,
  title         = {A Reproducible Universal Dependencies-Style Pipeline for
                   Katharevousa Greek Parliamentary Text},
  author        = {Mikros, George and Fitsilis, Fotios},
  year          = {2026},
  eprint        = {2605.22978},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2605.22978}
}
```
"""


def main() -> None:
    if not SRC_DIR.exists():
        raise SystemExit(f"Source model dir not found: {SRC_DIR}")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    # 1. Copy encoder/ and tokenizer/ verbatim (already HF-format).
    shutil.copytree(SRC_DIR / "encoder", OUT_DIR / "encoder")
    shutil.copytree(SRC_DIR / "tokenizer", OUT_DIR / "tokenizer")

    # 2. Strip encoder weights from parser_heads.pt to ship only heads.
    full_state = torch.load(SRC_DIR / "parser_heads.pt", map_location="cpu")
    head_state = {k: v for k, v in full_state.items() if not k.startswith("encoder.")}
    encoder_only_keys = sum(1 for k in full_state if k.startswith("encoder."))
    head_keys = len(head_state)
    print(
        f"parser_heads.pt: kept {head_keys} head tensors, "
        f"dropped {encoder_only_keys} encoder tensors."
    )
    torch.save(head_state, OUT_DIR / "parser_heads.pt")

    # 3. Copy metadata.json (label maps + training config).
    shutil.copy(SRC_DIR / "metadata.json", OUT_DIR / "metadata.json")

    # 4. Write model card.
    (OUT_DIR / "README.md").write_text(MODEL_CARD, encoding="utf-8")

    # 5. Report sizes.
    total_bytes = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file())
    print(f"Total release size: {total_bytes / 1e9:.2f} GB")
    print(f"Wrote release bundle to {OUT_DIR}")


if __name__ == "__main__":
    main()
