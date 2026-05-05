from __future__ import annotations



import argparse

import json

import random

from concurrent.futures import ThreadPoolExecutor

from pathlib import Path



from kathnlp.pipelines.annotators import (

    build_anthropic_annotator,

    build_gemini_annotator,

    build_openai_annotator,

)

from kathnlp.pipelines.corpus import SentenceRecord, extract_answer_files_sentences

from kathnlp.pipelines.serialization import write_conllu, write_sidecar_json

from kathnlp.pipelines.voting import vote_on_three_annotations

from kathnlp.schema import UDSentenceAnnotation





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description="Create majority-vote pilot dataset for Katharevousa NLP."

    )

    parser.add_argument(

        "--csv-paths",

        nargs="+",

        type=Path,

        required=True,

    )

    parser.add_argument("--sample-size", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=Path("data/processed/pilot"),

    )

    parser.add_argument(

        "--answer-column-name",

        default="Answer Files",

    )

    parser.add_argument(

        "--resume",

        action="store_true",

        help="Resume from checkpoint files in output-dir if available.",

    )

    parser.add_argument(

        "--checkpoint-every",

        type=int,

        default=5,

        help="Persist progress every N processed records.",

    )

    parser.add_argument(

        "--parallel-annotators",

        action="store_true",

        help="Annotate each sentence with 3 models in parallel (faster, less stable).",

    )

    return parser.parse_args()





def _sample_records(records: list[SentenceRecord], sample_size: int, seed: int) -> list[SentenceRecord]:

    random.seed(seed)

    if sample_size >= len(records):

        return records

    return random.sample(records, sample_size)





def _annotate_three(

    sentence_id: str,

    text: str,

    openai_ann,

    anthropic_ann,

    gemini_ann,

):

    with ThreadPoolExecutor(max_workers=3) as pool:

        f1 = pool.submit(openai_ann.annotate, sentence_id, text)

        f2 = pool.submit(anthropic_ann.annotate, sentence_id, text)

        f3 = pool.submit(gemini_ann.annotate, sentence_id, text)

        return f1.result(), f2.result(), f3.result()





def main() -> None:

    args = parse_args()

    records = extract_answer_files_sentences(

        args.csv_paths,

        answer_column_name=args.answer_column_name,

    )

    sampled = _sample_records(records, args.sample_size, args.seed)

    output_dir: Path = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)



    openai_ann = build_openai_annotator()

    anthropic_ann = build_anthropic_annotator()

    gemini_ann = build_gemini_annotator()



    auto_accepted: list[UDSentenceAnnotation] = []

    disagreements: list[dict] = []

    failures: list[dict] = []



    checkpoint_auto = output_dir / "checkpoint_auto_accepted.conllu"

    checkpoint_sidecar = output_dir / "checkpoint_auto_accepted.sidecar.json"

    checkpoint_disagreements = output_dir / "checkpoint_disagreements.json"

    checkpoint_failures = output_dir / "checkpoint_failures.json"

    completed_ids: set[str] = set()



    if args.resume and checkpoint_disagreements.exists():

        prior_disagreements = json.loads(checkpoint_disagreements.read_text(encoding="utf-8"))

        disagreements.extend(prior_disagreements)

        completed_ids.update(item["sent_id"] for item in prior_disagreements)

    if args.resume and checkpoint_failures.exists():

        prior_failures = json.loads(checkpoint_failures.read_text(encoding="utf-8"))

        failures.extend(prior_failures)

        completed_ids.update(item["sent_id"] for item in prior_failures)

    if args.resume and checkpoint_auto.exists():

        from kathnlp.training.dataset import load_conllu



        prior_auto = load_conllu(checkpoint_auto)

        auto_accepted.extend(prior_auto)

        completed_ids.update(item.sentence_id for item in prior_auto)



    for i, record in enumerate(sampled, start=1):

        if record.sentence_id in completed_ids:

            continue

        try:

            if args.parallel_annotators:

                ann_a, ann_b, ann_c = _annotate_three(

                    sentence_id=record.sentence_id,

                    text=record.text,

                    openai_ann=openai_ann,

                    anthropic_ann=anthropic_ann,

                    gemini_ann=gemini_ann,

                )

            else:

                ann_a = openai_ann.annotate(record.sentence_id, record.text)

                ann_b = anthropic_ann.annotate(record.sentence_id, record.text)

                ann_c = gemini_ann.annotate(record.sentence_id, record.text)

            vote = vote_on_three_annotations(ann_a, ann_b, ann_c)

            if vote.auto_accepted is not None:

                vote.auto_accepted.metadata["source_sheet"] = record.source_sheet

                vote.auto_accepted.metadata["source_row"] = str(record.source_row)

                auto_accepted.append(vote.auto_accepted)

            else:

                disagreements.append(

                    {

                        "sent_id": record.sentence_id,

                        "text": record.text,

                        "source_sheet": record.source_sheet,

                        "source_row": record.source_row,

                        "disagreement": vote.disagreement_payload,

                    }

                )

        except Exception as exc:

            failures.append(

                {

                    "sent_id": record.sentence_id,

                    "text": record.text[:500],

                    "source_sheet": record.source_sheet,

                    "source_row": record.source_row,

                    "error_type": type(exc).__name__,

                    "error": str(exc),

                }

            )

        print(

            json.dumps(

                {

                    "progress": i,

                    "sample_size": len(sampled),

                    "auto_accepted": len(auto_accepted),

                    "disagreements": len(disagreements),

                    "failures": len(failures),

                },

                ensure_ascii=False,

            )

        )

        if i % max(1, args.checkpoint_every) == 0:

            write_conllu(auto_accepted, checkpoint_auto)

            write_sidecar_json(auto_accepted, checkpoint_sidecar)

            checkpoint_disagreements.write_text(

                json.dumps(disagreements, ensure_ascii=False, indent=2),

                encoding="utf-8",

            )

            checkpoint_failures.write_text(

                json.dumps(failures, ensure_ascii=False, indent=2),

                encoding="utf-8",

            )



    write_conllu(auto_accepted, output_dir / "auto_accepted.conllu")

    write_sidecar_json(auto_accepted, output_dir / "auto_accepted.sidecar.json")

    (output_dir / "disagreements.json").write_text(

        json.dumps(disagreements, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    (output_dir / "failures.json").write_text(

        json.dumps(failures, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )



    summary = {

        "sample_size": len(sampled),

        "auto_accepted_sentences": len(auto_accepted),

        "disagreement_sentences": len(disagreements),

        "failed_sentences": len(failures),

        "auto_acceptance_rate": round(len(auto_accepted) / max(1, len(sampled)), 4),

        "annotators": [openai_ann.model_name, anthropic_ann.model_name, gemini_ann.model_name],

    }

    (output_dir / "pilot_summary.json").write_text(

        json.dumps(summary, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))





if __name__ == "__main__":

    main()

