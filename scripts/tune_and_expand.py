from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kathnlp.pipelines.corpus import extract_answer_files_sentences
from kathnlp.training.dataset import load_conllu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune active-learning thresholds and create scale-up batches.")
    parser.add_argument(
        "--csv-paths",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--pilot-dir",
        type=Path,
        default=Path("data/processed/pilot"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/scaleup"),
    )
    parser.add_argument("--next-batch-size", type=int, default=3000)
    return parser.parse_args()


def _uncertainty_score(text: str) -> float:
    archaic_markers = ["ὑπ", "ἐκ", "οἱ", "τοῦ", "διά", "περί", "ὅπως"]
    marker_hits = sum(1 for m in archaic_markers if m in text)
    length = len(text.split())
    punctuation = sum(1 for ch in text if ch in ",;:()[]")
    # More markers + long syntactic structures => harder annotation.
    return marker_hits * 1.5 + min(length / 20.0, 3.0) + min(punctuation / 10.0, 2.0)


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = extract_answer_files_sentences(args.csv_paths)
    pilot_gold = load_conllu(args.pilot_dir / "gold_provisional.conllu")
    pilot_ids = {s.sentence_id for s in pilot_gold}

    remaining = [r for r in all_records if r.sentence_id not in pilot_ids]
    scored = [
        {
            "sentence_id": r.sentence_id,
            "text": r.text,
            "source_sheet": r.source_sheet,
            "source_row": r.source_row,
            "uncertainty_score": round(_uncertainty_score(r.text), 4),
        }
        for r in remaining
    ]
    scored_df = pd.DataFrame(scored).sort_values("uncertainty_score", ascending=False)

    if scored_df.empty:
        raise ValueError("No remaining records found for expansion.")

    high_uncertainty_threshold = float(scored_df["uncertainty_score"].quantile(0.80))
    low_uncertainty_threshold = float(scored_df["uncertainty_score"].quantile(0.30))

    manual_priority = scored_df[scored_df["uncertainty_score"] >= high_uncertainty_threshold]
    auto_priority = scored_df[scored_df["uncertainty_score"] <= low_uncertainty_threshold]
    mixed_batch = pd.concat(
        [
            manual_priority.head(args.next_batch_size // 2),
            auto_priority.head(args.next_batch_size // 2),
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["sentence_id"])

    manual_priority.to_csv(output_dir / "manual_priority_pool.csv", index=False, encoding="utf-8-sig")
    auto_priority.to_csv(output_dir / "auto_priority_pool.csv", index=False, encoding="utf-8-sig")
    mixed_batch.to_csv(output_dir / "next_batch_mixed.csv", index=False, encoding="utf-8-sig")

    tuning_report = {
        "pilot_sentences": len(pilot_ids),
        "remaining_sentences": int(scored_df.shape[0]),
        "high_uncertainty_threshold": high_uncertainty_threshold,
        "low_uncertainty_threshold": low_uncertainty_threshold,
        "manual_priority_count": int(manual_priority.shape[0]),
        "auto_priority_count": int(auto_priority.shape[0]),
        "mixed_next_batch_count": int(mixed_batch.shape[0]),
        "recommendation": {
            "auto_accept_above_majority_confidence": "0.70",
            "send_to_human_if_any_field_disagrees": True,
            "reserve_manual_budget_for_top_uncertainty_percentile": "20%",
        },
    }

    (output_dir / "tuning_report.json").write_text(
        json.dumps(tuning_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(tuning_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
