from __future__ import annotations



import argparse

import json

import re

import subprocess

import sys

from datetime import datetime, timezone

from pathlib import Path



from kathnlp.pipelines.corpus import extract_answer_files_sentences





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description=(

            "Resume GPT gold-batch generation from the next unfinished offset. "

            "Safe to rerun after interruptions or reboot."

        )

    )

    parser.add_argument(

        "--csv-paths",

        nargs="+",

        type=Path,

        default=[

            Path("data/interim/reconstructed_sheets/1976_reconstructed.csv"),

            Path("data/interim/reconstructed_sheets/1977_reconstructed.csv"),

        ],

    )

    parser.add_argument(

        "--answer-column-name",

        default="Answer Files Reconstructed",

    )

    parser.add_argument(

        "--output-root",

        type=Path,

        default=Path("data/processed/gold_gpt5_batches"),

    )

    parser.add_argument(

        "--batch-size",

        type=int,

        default=50,

    )

    parser.add_argument(

        "--start-offset",

        type=int,

        default=None,

        help=(

            "Optional explicit resume anchor offset. "

            "If omitted, script auto-anchors to the first existing prod batch offset."

        ),

    )

    parser.add_argument(

        "--max-batches",

        type=int,

        default=1,

        help="How many new batches to run in this invocation.",

    )

    parser.add_argument(

        "--dry-run",

        action="store_true",

        help="Print next unfinished offset without launching a batch.",

    )

    parser.add_argument(

        "--batch-name-suffix",

        default="prod50",

    )

    parser.add_argument(

        "--checkpoint-every",

        type=int,

        default=10,

    )

    parser.add_argument(

        "--sleep-between-requests",

        type=float,

        default=1.2,

    )

    parser.add_argument(

        "--max-retries",

        type=int,

        default=2,

    )

    parser.add_argument(

        "--retry-backoff-seconds",

        type=float,

        default=10.0,

    )

    parser.add_argument(

        "--max-consecutive-429",

        type=int,

        default=8,

    )

    return parser.parse_args()





def _read_json(path: Path) -> dict:

    return json.loads(path.read_text(encoding="utf-8"))





def _scan_completed_offsets(output_root: Path, suffix: str) -> tuple[set[int], int]:

    output_root.mkdir(parents=True, exist_ok=True)

    completed_offsets: set[int] = set()

    max_batch_idx = 0

    batch_re = re.compile(rf"^batch_(\d+)_{re.escape(suffix)}$")



    for child in output_root.iterdir():

        if not child.is_dir():

            continue

        m = batch_re.match(child.name)

        if m:

            max_batch_idx = max(max_batch_idx, int(m.group(1)))

        summary_path = child / "gold_summary.json"

        if not summary_path.exists():

            continue

        try:

            summary = _read_json(summary_path)

        except Exception:

            continue

        processed = int(summary.get("processed_records", 0))

        offset = int(summary.get("offset", -1))

        if processed > 0 and offset >= 0:

            completed_offsets.add(offset)



    return completed_offsets, max_batch_idx





def _next_unfinished_offset(

    completed_offsets: set[int],

    total_records: int,

    start_offset: int,

    batch_size: int,

) -> int | None:

    for offset in range(max(0, start_offset), total_records, max(1, batch_size)):

        if offset not in completed_offsets:

            return offset

    return None





def _effective_start_offset(

    completed_offsets: set[int],

    requested_start_offset: int | None,

) -> int:

    if requested_start_offset is not None:

        return max(0, requested_start_offset)

    if completed_offsets:

        # Default to forward-only continuation from the latest completed offset.

        return max(0, max(completed_offsets))

    return 0





def _run_single_batch(args: argparse.Namespace, offset: int, batch_idx: int) -> int:

    batch_dir = args.output_root / f"batch_{batch_idx:03d}_{args.batch_name_suffix}"

    cmd = [

        sys.executable,

        "scripts/build_gpt_gold_dataset.py",

        "--csv-paths",

        *[str(p) for p in args.csv_paths],

        "--answer-column-name",

        args.answer_column_name,

        "--sample-size",

        str(args.batch_size),

        "--offset",

        str(offset),

        "--output-dir",

        str(batch_dir),

        "--checkpoint-every",

        str(args.checkpoint_every),

        "--sleep-between-requests",

        str(args.sleep_between_requests),

        "--max-retries",

        str(args.max_retries),

        "--retry-backoff-seconds",

        str(args.retry_backoff_seconds),

        "--max-consecutive-429",

        str(args.max_consecutive_429),

    ]

    print(

        json.dumps(

            {

                "event": "starting_batch",

                "batch_dir": str(batch_dir),

                "offset": offset,

                "batch_size": args.batch_size,

            },

            ensure_ascii=False,

        )

    )

    return subprocess.run(cmd, check=False).returncode





def _write_state(

    output_root: Path,

    *,

    last_attempted_offset: int | None,

    batches_launched: int,

    completed_offsets_count: int,

) -> None:

    state = {

        "updated_at_utc": datetime.now(timezone.utc).isoformat(),

        "last_attempted_offset": last_attempted_offset,

        "batches_launched_this_run": batches_launched,

        "completed_offsets_count": completed_offsets_count,

    }

    (output_root / "resume_state.json").write_text(

        json.dumps(state, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )





def main() -> None:

    args = parse_args()

    records = extract_answer_files_sentences(

        args.csv_paths,

        answer_column_name=args.answer_column_name,

    )

    total_records = len(records)

    completed_offsets, max_batch_idx = _scan_completed_offsets(

        args.output_root,

        args.batch_name_suffix,

    )

    start_anchor = _effective_start_offset(completed_offsets, args.start_offset)



    launched = 0

    last_attempted_offset: int | None = None



    target_batches = max(0, args.max_batches)

    if target_batches == 0 and not args.dry_run:

        print(

            json.dumps(

                {

                    "event": "no_op",

                    "message": "max-batches is 0; nothing launched.",

                },

                ensure_ascii=False,

            )

        )

        _write_state(

            args.output_root,

            last_attempted_offset=None,

            batches_launched=0,

            completed_offsets_count=len(completed_offsets),

        )

        return



    while launched < (1 if args.dry_run else target_batches):

        next_offset = _next_unfinished_offset(

            completed_offsets,

            total_records,

            start_anchor,

            args.batch_size,

        )

        if next_offset is None:

            print(

                json.dumps(

                    {

                        "event": "complete",

                        "message": "No unfinished offsets found.",

                        "total_records": total_records,

                    },

                    ensure_ascii=False,

                )

            )

            break



        if args.dry_run:

            print(

                json.dumps(

                    {

                        "event": "dry_run_next_offset",

                        "offset": next_offset,

                        "start_anchor": start_anchor,

                        "completed_offsets_count": len(completed_offsets),

                    },

                    ensure_ascii=False,

                )

            )

            break



        max_batch_idx += 1

        last_attempted_offset = next_offset

        code = _run_single_batch(args, next_offset, max_batch_idx)

        launched += 1



        if code != 0:

            print(

                json.dumps(

                    {

                        "event": "batch_failed",

                        "offset": next_offset,

                        "exit_code": code,

                    },

                    ensure_ascii=False,

                )

            )

            break



        completed_offsets, _ = _scan_completed_offsets(

            args.output_root,

            args.batch_name_suffix,

        )



    _write_state(

        args.output_root,

        last_attempted_offset=last_attempted_offset,

        batches_launched=launched,

        completed_offsets_count=len(completed_offsets),

    )





if __name__ == "__main__":

    main()

