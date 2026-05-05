from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from kathnlp.reconstruction import ReconstructionStats, reconstruct_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct wrapped/hyphenated OCR text in Answer Files columns."
    )
    parser.add_argument("--csv-paths", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--answer-column-name",
        default="Answer Files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/interim/reconstructed_sheets"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = ReconstructionStats()
    audit_rows: list[dict] = []

    for csv_path in args.csv_paths:
        frame = pd.read_csv(csv_path)
        if args.answer_column_name not in frame.columns:
            continue

        reconstructed_values: list[str] = []
        for idx, raw_text in frame[args.answer_column_name].fillna("").astype(str).items():
            stats.rows_total += 1
            reconstructed, details = reconstruct_text(raw_text)
            reconstructed_values.append(reconstructed)
            stats.join_repairs += int(details["join_repairs"])
            stats.newline_repairs += int(details["newline_repairs"])
            if details["changed"]:
                stats.rows_changed += 1
                audit_rows.append(
                    {
                        "sheet": csv_path.stem,
                        "row": idx + 1,
                        "join_repairs": details["join_repairs"],
                        "newline_repairs": details["newline_repairs"],
                        "original_preview": raw_text[:220],
                        "reconstructed_preview": reconstructed[:220],
                    }
                )

        frame["Answer Files Reconstructed"] = reconstructed_values
        out_path = output_dir / f"{csv_path.stem}_reconstructed.csv"
        frame.to_csv(out_path, index=False, encoding="utf-8-sig")

    (output_dir / "reconstruction_audit.json").write_text(
        json.dumps(audit_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "reconstruction_summary.json").write_text(
        json.dumps(stats.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
