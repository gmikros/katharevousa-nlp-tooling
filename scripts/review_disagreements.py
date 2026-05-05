from __future__ import annotations



import argparse

import json

from pathlib import Path





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description="Interactive reviewer for disagreement cases."

    )

    parser.add_argument(

        "--pilot-dir",

        type=Path,

        default=Path("data/processed/pilot"),

    )

    parser.add_argument(

        "--max-cases",

        type=int,

        default=100,

    )

    parser.add_argument(

        "--start-index",

        type=int,

        default=0,

    )

    return parser.parse_args()





def _preview_disagreements(disagreement: dict) -> str:

    diffs = disagreement.get("disagreements", [])

    if not diffs:

        return "No detailed field disagreements."

    lines = []

    for item in diffs[:5]:

        token_id = item.get("token_id", "?")

        form = item.get("form", "")

        vals = item.get("values", {})

        upos = vals.get("upos", [])

        lines.append(f"token={token_id}:{form} UPOS={upos}")

    if len(diffs) > 5:

        lines.append(f"... and {len(diffs) - 5} more token disagreements")

    return "\n".join(lines)





def main() -> None:

    args = parse_args()

    pilot_dir: Path = args.pilot_dir

    disagreements_path = pilot_dir / "disagreements.json"

    if not disagreements_path.exists():

        raise FileNotFoundError("Missing disagreements.json. Run build_pilot_dataset.py first.")



    disagreements = json.loads(disagreements_path.read_text(encoding="utf-8"))

    output_path = pilot_dir / "human_decisions.json"

    decisions: dict[str, str] = {}

    if output_path.exists():

        decisions = json.loads(output_path.read_text(encoding="utf-8"))



    end_index = min(len(disagreements), args.start_index + args.max_cases)

    previous_preview: str | None = None

    for i in range(args.start_index, end_index):

        item = disagreements[i]

        sent_id = item["sent_id"]

        if sent_id in decisions:

            continue

        disagreement = item.get("disagreement", {})

        candidates = disagreement.get("candidates", [])

        if len(candidates) != 3:

            continue



        print("\n" + "=" * 80)

        print(f"[{i+1}/{len(disagreements)}] sent_id={sent_id}")

        print(f"Text: {item.get('text', '')[:300]}")

        preview = _preview_disagreements(disagreement)

        print("Disagreement summary:")

        print(preview)

        if previous_preview is not None and preview == previous_preview:

            print(

                "WARNING: Preview is byte-identical to previous case. "

                "Re-check candidate CoNLL-U before adjudicating."

            )

        previous_preview = preview



        if any("heuristic" in str(c.get("model", "")) for c in candidates):

            print(

                "WARNING: These candidates are from heuristic fallback annotators, "

                "not live API model outputs."

            )

        print("\nChoose winning model:")

        print(f"  1) {candidates[0].get('model')}")

        print(f"  2) {candidates[1].get('model')}")

        print(f"  3) {candidates[2].get('model')}")

        print("  s) skip")

        print("  q) quit and save")



        choice = input("Your choice: ").strip().lower()

        if choice == "q":

            break

        if choice == "s" or choice == "":

            continue

        if choice in {"1", "2", "3"}:

            chosen = candidates[int(choice) - 1].get("model")

            decisions[sent_id] = chosen

            output_path.write_text(

                json.dumps(decisions, ensure_ascii=False, indent=2),

                encoding="utf-8",

            )

            print(f"Saved: {sent_id} -> {chosen}")



    output_path.write_text(

        json.dumps(decisions, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    print(f"\nSaved {len(decisions)} decisions to {output_path}")





if __name__ == "__main__":

    main()

