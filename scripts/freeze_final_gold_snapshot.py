from __future__ import annotations

import argparse
import json
from pathlib import Path

from kathnlp.training.dataset import load_conllu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze deduplicated final gold snapshot from batches + retries."
    )
    parser.add_argument(
        "--batches-root",
        type=Path,
        default=Path("data/processed/gold_gpt5_batches"),
    )
    parser.add_argument(
        "--retries-root",
        type=Path,
        default=Path("data/processed/gold_gpt5_retries"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/final_gold"),
    )
    return parser.parse_args()


def _collect_conllu_files(root: Path, pattern: str) -> list[Path]:
    if not root.exists():
        return []
    return sorted(root.glob(pattern))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batch_files = _collect_conllu_files(args.batches_root, "batch_*/gold_gpt.conllu")
    retry_files = _collect_conllu_files(args.retries_root, "retry_*/gold_gpt.conllu")

    # Priority: first-pass batches, then retries overwrite same sent_id if present.
    ordered_sources = [("batch", p) for p in batch_files] + [
        ("retry", p) for p in retry_files
    ]

    by_sent_id: dict[str, tuple[str, Path, object]] = {}
    for source_kind, conllu_path in ordered_sources:
        for sentence in load_conllu(conllu_path):
            by_sent_id[sentence.sentence_id] = (source_kind, conllu_path, sentence)

    final_sentences = [entry[2] for _, entry in sorted(by_sent_id.items(), key=lambda x: x[0])]
    final_conllu = args.output_dir / "gold_final.conllu"
    final_conllu.write_text(
        "\n\n".join(sentence.to_conllu() for sentence in final_sentences) + "\n",
        encoding="utf-8",
    )

    source_breakdown = {"batch": 0, "retry": 0}
    for source_kind, _, _ in by_sent_id.values():
        source_breakdown[source_kind] += 1

    manifest = {
        "final_sentences": len(final_sentences),
        "source_breakdown": source_breakdown,
        "batch_files_used": [str(p) for p in batch_files],
        "retry_files_used": [str(p) for p in retry_files],
        "output_conllu": str(final_conllu),
    }
    (args.output_dir / "snapshot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
