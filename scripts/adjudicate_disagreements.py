from __future__ import annotations



import argparse

import json

from pathlib import Path



from kathnlp.pipelines.serialization import parse_conllu_sentence, write_conllu, write_sidecar_json

from kathnlp.schema import UDSentenceAnnotation





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Resolve disagreement queue into pilot gold.")

    parser.add_argument(

        "--pilot-dir",

        type=Path,

        default=Path("data/processed/pilot"),

    )

    parser.add_argument(

        "--strategy",

        choices=["openai_first", "anthropic_first", "gemini_first"],

        default="openai_first",

    )

    parser.add_argument(

        "--decisions-file",

        type=Path,

        default=None,

        help="Optional JSON file with sent_id -> chosen_model decisions.",

    )

    return parser.parse_args()





def _model_index(strategy: str) -> int:

    if strategy == "openai_first":

        return 0

    if strategy == "anthropic_first":

        return 1

    return 2





def main() -> None:

    args = parse_args()

    pilot_dir: Path = args.pilot_dir

    auto_sidecar = pilot_dir / "auto_accepted.sidecar.json"

    disagreements_path = pilot_dir / "disagreements.json"

    auto_conllu = pilot_dir / "auto_accepted.conllu"



    if not auto_sidecar.exists() or not disagreements_path.exists() or not auto_conllu.exists():

        raise FileNotFoundError("Pilot artifacts not found. Run build_pilot_dataset.py first.")



    auto_sentences: list[UDSentenceAnnotation] = []

    blocks = auto_conllu.read_text(encoding="utf-8").strip().split("\n\n")

    for block in blocks:

        if block.strip():

            auto_sentences.append(parse_conllu_sentence(block))



    disagreements = json.loads(disagreements_path.read_text(encoding="utf-8"))

    picked_sentences: list[UDSentenceAnnotation] = []

    adjudication_log: list[dict] = []



    idx = _model_index(args.strategy)

    user_decisions: dict[str, str] = {}

    if args.decisions_file is not None and args.decisions_file.exists():

        user_decisions = json.loads(args.decisions_file.read_text(encoding="utf-8"))



    for item in disagreements:

        candidates = item["disagreement"]["candidates"]

        model_to_candidate = {c.get("model"): c for c in candidates}

        decision_model = user_decisions.get(item["sent_id"])

        if decision_model in model_to_candidate:

            chosen = model_to_candidate[decision_model]

            decision_source = "human_decision_file"

        else:

            chosen = candidates[idx]

            decision_source = "strategy_default"

        sentence = parse_conllu_sentence(chosen["conllu"])

        sentence.metadata["adjudication_strategy"] = args.strategy

        sentence.metadata["adjudication_needs_human_review"] = (

            "false" if decision_source == "human_decision_file" else "true"

        )

        picked_sentences.append(sentence)

        adjudication_log.append(

            {

                "sent_id": item["sent_id"],

                "strategy_choice": args.strategy,

                "chosen_model": chosen.get("model"),

                "decision_source": decision_source,

                "needs_human_review": decision_source != "human_decision_file",

            }

        )



    all_sentences = auto_sentences + picked_sentences

    all_sentences.sort(key=lambda s: s.sentence_id)

    write_conllu(all_sentences, pilot_dir / "gold_provisional.conllu")

    write_sidecar_json(all_sentences, pilot_dir / "gold_provisional.sidecar.json")

    (pilot_dir / "adjudication_log.json").write_text(

        json.dumps(adjudication_log, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    print(

        json.dumps(

            {

                "total_gold_sentences": len(all_sentences),

                "auto_accepted": len(auto_sentences),

                "adjudicated_by_strategy": len(picked_sentences),

                "strategy": args.strategy,

            },

            ensure_ascii=False,

            indent=2,

        )

    )





if __name__ == "__main__":

    main()

