from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from kathnlp.pipelines.annotators import build_openai_annotator
from kathnlp.pipelines.corpus import SentenceRecord, extract_answer_files_sentences
from kathnlp.pipelines.serialization import write_conllu, write_sidecar_json
from kathnlp.schema import UDSentenceAnnotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create GPT-only gold dataset from reconstructed text."
    )
    parser.add_argument("--csv-paths", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--answer-column-name",
        default="Answer Files Reconstructed",
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index in deterministic sentence stream for batching.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/gold_gpt5"),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max per-sentence retries on transient API failures.",
    )
    parser.add_argument(
        "--retry-backoff-seconds",
        type=float,
        default=8.0,
        help="Base backoff multiplier for retries (attempt * value).",
    )
    parser.add_argument(
        "--sleep-between-requests",
        type=float,
        default=0.5,
        help="Delay between sentence requests to reduce throttling.",
    )
    parser.add_argument(
        "--max-consecutive-429",
        type=int,
        default=12,
        help="Abort batch early if this many consecutive 429 failures occur.",
    )
    return parser.parse_args()


def _slice_batch(
    records: list[SentenceRecord],
    sample_size: int,
    offset: int,
) -> list[SentenceRecord]:
    ordered = sorted(records, key=lambda r: r.sentence_id)
    start = max(0, offset)
    end = min(len(ordered), start + sample_size)
    return ordered[start:end]


def _is_parse_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    msg = str(exc).lower()
    return (
        type(exc).__name__ in {"SyntaxError", "JSONDecodeError"}
        or "invalid syntax" in msg
        or "json" in msg
    )


ENUM_MARKER_RE = re.compile(r"(?:(?<=\s)|^)(?:\d{1,2}[\)\.]|[-–—]|[α-ωΑ-Ω][\)\.])\s+")


def _split_enumerative_record(record: SentenceRecord) -> list[SentenceRecord]:
    text = (record.text or "").strip()
    words = text.split()
    if len(words) <= 70 and len(text) <= 360:
        return [record]

    has_enum = bool(ENUM_MARKER_RE.search(text)) or ":" in text or ";" in text
    if not has_enum:
        return [record]

    parts: list[str] = []
    coarse = [p.strip() for p in re.split(r"(?<=[;·])\s+|\n+", text) if p.strip()]
    for chunk in coarse:
        markers = [m.start() for m in ENUM_MARKER_RE.finditer(chunk)]
        if len(chunk.split()) <= 70 or len(markers) < 2:
            parts.append(chunk)
            continue
        markers.append(len(chunk))
        for i in range(len(markers) - 1):
            seg = chunk[markers[i] : markers[i + 1]].strip(" -–—")
            if seg:
                parts.append(seg)

    normalized: list[str] = []
    for part in parts:
        part_words = part.split()
        if len(part_words) <= 70:
            normalized.append(part)
            continue
        for i in range(0, len(part_words), 55):
            piece = " ".join(part_words[i : i + 55]).strip()
            if piece:
                normalized.append(piece)

    cleaned = [p for p in normalized if len(p) >= 18]
    if len(cleaned) <= 1:
        return [record]

    subrecords: list[SentenceRecord] = []
    for idx, part in enumerate(cleaned, start=1):
        subrecords.append(
            SentenceRecord(
                sentence_id=f"{record.sentence_id}-p{idx}",
                text=part,
                source_sheet=record.source_sheet,
                source_row=record.source_row,
            )
        )
    return subrecords


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    records = extract_answer_files_sentences(
        args.csv_paths,
        answer_column_name=args.answer_column_name,
    )
    sampled = _slice_batch(records, args.sample_size, args.offset)

    annotator = build_openai_annotator()
    fallback_max_tokens = int(os.getenv("OPENAI_FALLBACK_MAX_COMPLETION_TOKENS", "2600"))
    strict_fallback_annotator = build_openai_annotator(
        strict_json=True,
        max_completion_tokens_override=fallback_max_tokens,
    )
    accepted: list[UDSentenceAnnotation] = []
    failures: list[dict] = []
    consecutive_429 = 0
    processed_records = 0
    halted_on_rate_limit = False
    recovered_by_strict_fallback = 0
    split_expansions = 0
    annotation_units = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for i, record in enumerate(sampled, start=1):
        processed_records = i
        work_items = _split_enumerative_record(record)
        if len(work_items) > 1:
            split_expansions += len(work_items) - 1

        for work_record in work_items:
            annotation_units += 1
            if args.sleep_between_requests > 0:
                time.sleep(args.sleep_between_requests)

            attempts = 0
            last_exc: Exception | None = None
            ann = None
            while attempts < max(1, args.max_retries):
                attempts += 1
                try:
                    ann = annotator.annotate(work_record.sentence_id, work_record.text)
                    ann.validate()
                    break
                except Exception as exc:
                    last_exc = exc
                    msg = str(exc)
                    transient_parse_error = _is_parse_error(exc)
                    if (
                        ("429" in msg or transient_parse_error)
                        and attempts < max(1, args.max_retries)
                    ):
                        time.sleep(max(0.0, args.retry_backoff_seconds) * attempts)
                        continue
                    ann = None
                    break

            if ann is None and _is_parse_error(last_exc):
                strict_exc: Exception | None = None
                try:
                    ann = strict_fallback_annotator.annotate(
                        work_record.sentence_id, work_record.text
                    )
                    ann.validate()
                    ann.metadata["fallback_mode"] = "strict_json"
                    recovered_by_strict_fallback += 1
                except Exception as exc:
                    strict_exc = exc
                    ann = None
                if ann is None and strict_exc is not None:
                    last_exc = strict_exc

            if ann is not None:
                consecutive_429 = 0
                ann.metadata["source_sheet"] = work_record.source_sheet
                ann.metadata["source_row"] = str(work_record.source_row)
                ann.metadata["gold_source"] = "gpt_primary"
                if work_record.sentence_id != record.sentence_id:
                    ann.metadata["split_from_sent_id"] = record.sentence_id
                total_prompt_tokens += int(ann.metadata.get("prompt_tokens", "0"))
                total_completion_tokens += int(ann.metadata.get("completion_tokens", "0"))
                total_tokens += int(ann.metadata.get("total_tokens", "0"))
                accepted.append(ann)
            else:
                exc = last_exc if last_exc is not None else Exception("Unknown annotation error")
                if "429" in str(exc):
                    consecutive_429 += 1
                else:
                    consecutive_429 = 0
                failures.append(
                    {
                        "sent_id": work_record.sentence_id,
                        "source_sheet": work_record.source_sheet,
                        "source_row": work_record.source_row,
                        "text": work_record.text[:500],
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "attempts": attempts,
                    }
                )
                if consecutive_429 >= max(1, args.max_consecutive_429):
                    halted_on_rate_limit = True
                    print(
                        json.dumps(
                            {
                                "event": "halted_on_rate_limit",
                                "progress": i,
                                "consecutive_429": consecutive_429,
                            },
                            ensure_ascii=False,
                        )
                    )
                    break

        if halted_on_rate_limit:
            break

        if i % max(1, args.checkpoint_every) == 0:
            write_conllu(accepted, output_dir / "checkpoint_gold_gpt.conllu")
            write_sidecar_json(accepted, output_dir / "checkpoint_gold_gpt.sidecar.json")
            (output_dir / "checkpoint_failures.json").write_text(
                json.dumps(failures, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "progress": i,
                        "sample_size": len(sampled),
                        "annotation_units": annotation_units,
                        "accepted": len(accepted),
                        "failures": len(failures),
                    },
                    ensure_ascii=False,
                )
            )

    write_conllu(accepted, output_dir / "gold_gpt.conllu")
    write_sidecar_json(accepted, output_dir / "gold_gpt.sidecar.json")
    (output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "sample_size": len(sampled),
        "annotation_units": annotation_units,
        "split_expansions": split_expansions,
        "processed_records": processed_records,
        "offset": args.offset,
        "accepted_sentences": len(accepted),
        "failed_sentences": len(failures),
        "success_rate": round(len(accepted) / max(1, annotation_units), 4),
        "annotator": annotator.model_name,
        "halted_on_rate_limit": halted_on_rate_limit,
        "recovered_by_strict_fallback": recovered_by_strict_fallback,
        "token_usage": {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_total_tokens_per_accepted": round(
                total_tokens / max(1, len(accepted)),
                2,
            ),
        },
    }
    (output_dir / "gold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
