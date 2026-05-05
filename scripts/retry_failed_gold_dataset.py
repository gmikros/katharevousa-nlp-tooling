from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from kathnlp.pipelines.annotators import build_openai_annotator
from kathnlp.pipelines.corpus import SentenceRecord, extract_answer_files_sentences
from kathnlp.pipelines.serialization import write_conllu, write_sidecar_json
from kathnlp.schema import UD_DEPREL, UD_FEAT_KEYS, UDSentenceAnnotation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry unresolved sentence IDs from GPT gold batches."
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
        "--batches-root",
        type=Path,
        default=Path("data/processed/gold_gpt5_batches"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/gold_gpt5_retries/retry_001"),
    )
    parser.add_argument("--max-items", type=int, default=50)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=10.0)
    parser.add_argument("--sleep-between-requests", type=float, default=1.0)
    return parser.parse_args()


def _accepted_ids_from_conllu(dir_root: Path) -> set[str]:
    accepted: set[str] = set()
    for conllu in dir_root.glob("**/gold_gpt.conllu"):
        text = conllu.read_text(encoding="utf-8")
        for line in text.splitlines():
            if line.startswith("# sent_id = "):
                accepted.add(line.split("=", 1)[1].strip())
    return accepted


def _is_parse_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    msg = str(exc).lower()
    return (
        type(exc).__name__ in {"SyntaxError", "JSONDecodeError"}
        or "invalid syntax" in msg
        or "json" in msg
    )


def _repair_annotation(ann: UDSentenceAnnotation) -> UDSentenceAnnotation:
    token_ids = {t.id for t in ann.tokens}
    if not token_ids:
        return ann

    # Clamp FEATS / DEPREL to schema before validation.
    for tok in ann.tokens:
        tok.feats = {k: v for k, v in tok.feats.items() if k in UD_FEAT_KEYS}
        base = tok.deprel.split(":", 1)[0]
        if base not in UD_DEPREL:
            tok.deprel = "dep"
        if tok.head not in token_ids and tok.head != 0:
            tok.head = 0

    roots = [t for t in ann.tokens if t.head == 0]
    if not roots:
        root = min(ann.tokens, key=lambda t: t.id)
        root.head = 0
        root.deprel = "root"
        roots = [root]

    primary_root = min(roots, key=lambda t: t.id)
    primary_root.head = 0
    primary_root.deprel = "root"
    for tok in roots:
        if tok.id == primary_root.id:
            continue
        tok.head = primary_root.id
        tok.deprel = "conj"

    return ann


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = extract_answer_files_sentences(
        args.csv_paths,
        answer_column_name=args.answer_column_name,
    )
    record_map: dict[str, SentenceRecord] = {r.sentence_id: r for r in records}

    accepted_ids = _accepted_ids_from_conllu(args.batches_root)
    unresolved_ids = sorted(set(record_map.keys()) - accepted_ids)
    selected_ids = unresolved_ids[args.offset : args.offset + max(1, args.max_items)]
    selected_records = [record_map[sid] for sid in selected_ids]

    annotator = build_openai_annotator()
    strict_fallback = build_openai_annotator(
        strict_json=True,
        max_completion_tokens_override=int(
            os.getenv("OPENAI_FALLBACK_MAX_COMPLETION_TOKENS", "2800")
        ),
    )

    accepted: list[UDSentenceAnnotation] = []
    failures: list[dict] = []
    repaired_successes = 0
    strict_recoveries = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    for i, record in enumerate(selected_records, start=1):
        if args.sleep_between_requests > 0:
            time.sleep(args.sleep_between_requests)

        attempts = 0
        last_exc: Exception | None = None
        ann: UDSentenceAnnotation | None = None

        while attempts < max(1, args.max_retries):
            attempts += 1
            try:
                ann = annotator.annotate(record.sentence_id, record.text)
                try:
                    ann.validate()
                except Exception:
                    ann = _repair_annotation(ann)
                    ann.validate()
                    repaired_successes += 1
                break
            except Exception as exc:
                last_exc = exc
                if _is_parse_error(exc) and attempts < max(1, args.max_retries):
                    time.sleep(max(0.0, args.retry_backoff_seconds) * attempts)
                    continue
                ann = None
                break

        if ann is None and _is_parse_error(last_exc):
            try:
                ann = strict_fallback.annotate(record.sentence_id, record.text)
                try:
                    ann.validate()
                except Exception:
                    ann = _repair_annotation(ann)
                    ann.validate()
                    repaired_successes += 1
                ann.metadata["fallback_mode"] = "strict_json"
                strict_recoveries += 1
            except Exception as exc:
                last_exc = exc
                ann = None

        if ann is not None:
            ann.metadata["source_sheet"] = record.source_sheet
            ann.metadata["source_row"] = str(record.source_row)
            ann.metadata["gold_source"] = "gpt_retry"
            total_prompt_tokens += int(ann.metadata.get("prompt_tokens", "0"))
            total_completion_tokens += int(ann.metadata.get("completion_tokens", "0"))
            total_tokens += int(ann.metadata.get("total_tokens", "0"))
            accepted.append(ann)
        else:
            exc = last_exc if last_exc is not None else Exception("Unknown annotation error")
            failures.append(
                {
                    "sent_id": record.sentence_id,
                    "source_sheet": record.source_sheet,
                    "source_row": record.source_row,
                    "text": record.text[:500],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "attempts": attempts,
                }
            )

        if i % 10 == 0:
            print(
                json.dumps(
                    {
                        "progress": i,
                        "sample_size": len(selected_records),
                        "accepted": len(accepted),
                        "failures": len(failures),
                    },
                    ensure_ascii=False,
                )
            )

    write_conllu(accepted, args.output_dir / "gold_gpt.conllu")
    write_sidecar_json(accepted, args.output_dir / "gold_gpt.sidecar.json")
    (args.output_dir / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "sample_size": len(selected_records),
        "offset": args.offset,
        "accepted_sentences": len(accepted),
        "failed_sentences": len(failures),
        "success_rate": round(len(accepted) / max(1, len(selected_records)), 4),
        "strict_recoveries": strict_recoveries,
        "repaired_successes": repaired_successes,
        "annotator": annotator.model_name,
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
    (args.output_dir / "gold_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
