"""Stage the Katharevousa NLP treebank for release on the Hugging Face Hub.

Produces a self-contained `dataset_release/` directory with:
  - train.conllu / test.conllu                 (raw UD CoNLL-U)
  - train.jsonl  / test.jsonl                  (HF Datasets Viewer friendly)
  - snapshot_manifest.json                     (provenance manifest)
  - split.json                                 (the seed-42 train/test split)
  - annotation_schema.yaml                     (schema sidecar)
  - reports/                                   (all benchmark JSONs)
  - README.md                                  (dataset card)

Run from the repository root:

    python scripts/build_dataset_release.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
GOLD_CONLLU = REPO_ROOT / "data" / "processed" / "final_gold" / "gold_final.conllu"
SPLIT_JSON = REPO_ROOT / "reports" / "transformer_parser_v1_opt_split.json"
MANIFEST = REPO_ROOT / "data" / "processed" / "final_gold" / "snapshot_manifest.json"
SCHEMA = REPO_ROOT / "configs" / "annotation_schema.yaml"
REPORTS_DIR = REPO_ROOT / "reports"
OUT_DIR = REPO_ROOT / "dataset_release"


def parse_conllu(text: str) -> List[Dict]:
    """Parse a CoNLL-U file into a list of sentence dicts."""
    sentences: List[Dict] = []
    current_comments: List[str] = []
    current_tokens: List[Dict] = []
    metadata: Dict[str, str] = {}

    def flush():
        if not current_tokens and not current_comments:
            return
        sent = {
            "sent_id": metadata.get("sent_id", ""),
            "text": metadata.get("text", ""),
            "annotator": metadata.get("annotator", ""),
            "source_sheet": metadata.get("source_sheet", ""),
            "source_row": _maybe_int(metadata.get("source_row", "")),
            "gold_source": metadata.get("gold_source", ""),
            "tokens": list(current_tokens),
            "comments": list(current_comments),
        }
        sentences.append(sent)

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if line.startswith("#"):
            current_comments.append(line)
            stripped = line.lstrip("#").strip()
            if "=" in stripped:
                key, _, value = stripped.partition("=")
                metadata[key.strip()] = value.strip()
            continue
        if not line.strip():
            flush()
            current_comments = []
            current_tokens = []
            metadata = {}
            continue
        fields = line.split("\t")
        if len(fields) < 10:
            continue
        tok_id = fields[0]
        if "-" in tok_id or "." in tok_id:
            # multiword token / empty node — keep as-is, parse head as None
            head = None
        else:
            head = _maybe_int(fields[6])
        current_tokens.append(
            {
                "id": tok_id,
                "form": fields[1],
                "lemma": fields[2],
                "upos": fields[3],
                "xpos": fields[4] if fields[4] != "_" else None,
                "feats": fields[5] if fields[5] != "_" else None,
                "head": head,
                "deprel": fields[7] if fields[7] != "_" else None,
                "deps": fields[8] if fields[8] != "_" else None,
                "misc": fields[9] if fields[9] != "_" else None,
            }
        )

    flush()
    return sentences


def _maybe_int(value: str):
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def render_conllu(sentence: Dict) -> str:
    """Render a parsed sentence back to CoNLL-U format."""
    lines = list(sentence["comments"])
    for tok in sentence["tokens"]:
        row = [
            str(tok["id"]),
            tok["form"],
            tok["lemma"],
            tok["upos"],
            tok["xpos"] or "_",
            tok["feats"] or "_",
            "_" if tok["head"] is None else str(tok["head"]),
            tok["deprel"] or "_",
            tok["deps"] or "_",
            tok["misc"] or "_",
        ]
        lines.append("\t".join(row))
    return "\n".join(lines) + "\n\n"


def write_conllu(path: Path, sentences: List[Dict]) -> None:
    path.write_text("".join(render_conllu(s) for s in sentences), encoding="utf-8")


def write_jsonl(path: Path, sentences: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as fp:
        for sent in sentences:
            row = {
                "sent_id": sent["sent_id"],
                "text": sent["text"],
                "annotator": sent["annotator"],
                "source_sheet": sent["source_sheet"],
                "source_row": sent["source_row"],
                "tokens": sent["tokens"],
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_split(sentences: List[Dict], split_payload: Dict) -> Tuple[List[Dict], List[Dict]]:
    train_ids = set(split_payload["train_sent_ids"])
    test_ids = set(split_payload["test_sent_ids"])
    train, test = [], []
    for sent in sentences:
        sid = sent["sent_id"]
        if sid in train_ids:
            train.append(sent)
        elif sid in test_ids:
            test.append(sent)
    return train, test


DATASET_CARD = """---
language:
  - el
license: cc-by-sa-4.0
task_categories:
  - token-classification
  - feature-extraction
tags:
  - katharevousa
  - greek
  - historical-greek
  - dependency-parsing
  - universal-dependencies
  - conllu
  - treebank
  - historical-nlp
  - parliamentary-corpora
  - low-resource
size_categories:
  - 1K<n<10K
pretty_name: Katharevousa NLP Treebank
arxiv: 2605.22978
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: test
        path: test.jsonl
---

# kathnlp Katharevousa Greek Treebank

A Universal-Dependencies-style reference treebank for **Katharevousa Greek**, the archaizing official register used in 20th-century Greek law, administration, and parliamentary discourse. The treebank covers 1,697 sentences from written parliamentary questions of the early Third Hellenic Republic (1976–1977) and is released alongside the [`kathnlp`](https://github.com/gmikros/katharevousa-nlp-tooling) parsing pipeline.

- **Paper:** [A Reproducible Universal Dependencies-Style Pipeline for Katharevousa Greek Parliamentary Text](https://arxiv.org/abs/2605.22978) (arXiv:2605.22978)
- **Code:** <https://github.com/gmikros/katharevousa-nlp-tooling>
- **Companion model:** [`gmikros/kathnlp-xlmr`](https://huggingface.co/gmikros/kathnlp-xlmr) (release candidate, planned)

## Dataset summary

| Item | Count |
|---|---|
| Frozen reference sentences | 1,697 |
| Training sentences | 1,357 |
| Held-out test sentences | 340 |
| Held-out test tokens | 4,093 |

The split uses **seed 42** and is identical to the one used in every benchmark row of the accompanying paper. Use it to reproduce the published numbers and to evaluate new models on a comparable surface.

## Files

| File | Purpose |
|---|---|
| `train.conllu` / `test.conllu` | Raw UD CoNLL-U format for traditional NLP tooling. |
| `train.jsonl` / `test.jsonl` | One sentence per line; structured tokens. Powers the HF Dataset Viewer. |
| `snapshot_manifest.json` | Provenance manifest of the merged batch and retry files. |
| `split.json` | The seed-42 train/test split (sentence IDs). |
| `annotation_schema.yaml` | Katharevousa-aware annotation schema (UD v2 + sidecar fields). |
| `reports/*.json` | Per-model benchmark reports (spaCy, Stanza, custom Stanza, mBERT, XLM-R, feature-based). |

## Schema

The CoNLL-U columns follow [UD v2](https://universaldependencies.org/format.html): `ID FORM LEMMA UPOS XPOS FEATS HEAD DEPREL DEPS MISC`. The JSONL rows expose:

```json
{
  "sent_id": "1977_reconstructed-...",
  "text": "Full sentence string.",
  "annotator": "gpt-5.5",
  "source_sheet": "1977_reconstructed",
  "source_row": 12,
  "tokens": [
    {"id": "1", "form": "Ἡ", "lemma": "ὁ", "upos": "DET", "feats": "Case=Nom|...", "head": 2, "deprel": "det", ...}
  ]
}
```

## Curation and provenance

- **Source archive.** Greek parliamentary written questions from 1976–1977, OCR-derived and reconstructed for word-internal hyphenation and line-break artifacts. The underlying documents were digitised from the historical archive of the 1st Parliamentary Term (1974–1977) of the Hellenic Parliament — 1,674 page images / 1,338 questions processed with a custom OCR platform (YOLOv5 text-line segmentation + Calamari-OCR), reaching 98.7% character recognition accuracy — by Fitsilis et al. (ICDAR 2024 Workshops; DOI [10.1007/978-3-031-70645-5_8](https://doi.org/10.1007/978-3-031-70645-5_8)). The same archive underlies the companion DSH digital-humanities study.
- **Annotation.** Schema-constrained LLM-assisted annotation (GPT-5 family) with structured JSON output, automatic validation, retry queues, and deterministic snapshotting. 1,565 sentences come from the main batch path; 132 are retry replacements that passed validation after the original batch failed.
- **Status.** The annotations are **automatically validated**, not fully expert-adjudicated. A philologist adjudication round is in progress and will be released as a versioned update.

## How to load

### Hugging Face `datasets`

```python
from datasets import load_dataset

ds = load_dataset("gmikros/kathnlp-treebank")
print(ds)
# DatasetDict({
#     train: Dataset({ features: [...], num_rows: 1357 }),
#     test:  Dataset({ features: [...], num_rows: 340  }),
# })
```

### Raw CoNLL-U

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="gmikros/kathnlp-treebank",
    filename="train.conllu",
    repo_type="dataset",
)
text = open(path, encoding="utf-8").read()
```

## Benchmark protocol

All published numbers in the accompanying paper use the **fixed seed-42 split** and identical scoring code (`src/kathnlp/evaluation/metrics.py` in the [GitHub repository](https://github.com/gmikros/katharevousa-nlp-tooling)). The per-model reports under `reports/` capture UPOS accuracy, weighted DEPREL F1, UAS, and LAS for spaCy Greek, Stanza Greek, Stanza Ancient Greek PROIEL, a custom-trained Stanza, mBERT, the XLM-R release candidate, and a feature-based logistic baseline.

## Limitations

- Small held-out split (340 sentences / 4,093 tokens).
- Genre limited to parliamentary questions; transfer to legal texts, decrees, or newspapers is untested.
- Automatically validated rather than expert-adjudicated annotations.
- Polytonic and monotonic orthographic variation is preserved as in the source archive.

## License

Released under **Creative Commons Attribution-ShareAlike 4.0** (`cc-by-sa-4.0`), in line with most Universal Dependencies treebanks. The source archival material is in the public domain (Greek parliamentary records).

## Citation

Please cite the accompanying paper:

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

The source corpus digitization is described in:

```bibtex
@inproceedings{fitsilis2024digitization,
  title     = {Digitization of Written Parliamentary Questions from the
               Historical Archive (1974--1977) of the Hellenic Parliament},
  author    = {Fitsilis, Fotios and Gatos, Basilis and Palaiologos, Konstantinos
               and Kaddas, Panagiotis and Kyrkos, Charalambis and
               Georgoulea, Maria-Eleni and Armenakis, Yiannis and Tasouli, Christina
               and Mikros, George and Rozenberg, Olivier and Kiousi, Eleni},
  booktitle = {Document Analysis and Recognition -- ICDAR 2024 Workshops},
  series    = {Lecture Notes in Computer Science},
  volume    = {14935},
  pages     = {103--117},
  year      = {2024},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-031-70645-5_8}
}
```
"""


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    text = GOLD_CONLLU.read_text(encoding="utf-8")
    sentences = parse_conllu(text)
    print(f"Parsed {len(sentences)} sentences from {GOLD_CONLLU.name}.")

    split_payload = json.loads(SPLIT_JSON.read_text(encoding="utf-8"))
    train, test = make_split(sentences, split_payload)
    print(f"Train sentences: {len(train)}; test sentences: {len(test)}.")

    write_conllu(OUT_DIR / "train.conllu", train)
    write_conllu(OUT_DIR / "test.conllu", test)
    write_jsonl(OUT_DIR / "train.jsonl", train)
    write_jsonl(OUT_DIR / "test.jsonl", test)

    shutil.copy(MANIFEST, OUT_DIR / "snapshot_manifest.json")
    shutil.copy(SPLIT_JSON, OUT_DIR / "split.json")
    shutil.copy(SCHEMA, OUT_DIR / "annotation_schema.yaml")

    reports_out = OUT_DIR / "reports"
    reports_out.mkdir()
    for report in REPORTS_DIR.glob("*.json"):
        shutil.copy(report, reports_out / report.name)

    (OUT_DIR / "README.md").write_text(DATASET_CARD, encoding="utf-8")

    test_token_count = sum(len(s["tokens"]) for s in test)
    print(f"Test token count (informational): {test_token_count}")
    print(f"Wrote release bundle to {OUT_DIR}")


if __name__ == "__main__":
    main()
