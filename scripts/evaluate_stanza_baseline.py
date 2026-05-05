from __future__ import annotations

import argparse
import json
from pathlib import Path

from kathnlp.evaluation.metrics import evaluate, evaluate_hybrid_proxy_baseline
from kathnlp.training.dataset import load_conllu, sentence_to_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a Stanza Greek pipeline on the fixed test split."
    )
    parser.add_argument(
        "--gold-path",
        type=Path,
        default=Path("data/processed/final_gold/gold_final.conllu"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("reports/transformer_parser_v1_opt_split.json"),
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=Path("reports/stanza_baseline_report.json"),
    )
    parser.add_argument("--lang", type=str, default="el")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_test_sentences(gold_path: Path, split_manifest: Path):
    all_sentences = load_conllu(gold_path)
    by_id = {s.sentence_id: s for s in all_sentences}
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    test_ids: list[str] = split["test_sent_ids"]
    test_sentences = [by_id[sent_id] for sent_id in test_ids if sent_id in by_id]
    missing_ids = [sent_id for sent_id in test_ids if sent_id not in by_id]
    return all_sentences, test_sentences, split, missing_ids


def main() -> None:
    args = parse_args()

    try:
        import stanza
    except ImportError as exc:
        raise SystemExit(
            "Stanza is not installed. Run: python -m pip install stanza"
        ) from exc

    all_sentences, test_sentences, split, missing_ids = _load_test_sentences(
        gold_path=args.gold_path,
        split_manifest=args.split_manifest,
    )

    stanza.download(args.lang, verbose=False)
    nlp = stanza.Pipeline(
        lang=args.lang,
        processors="tokenize,pos,lemma,depparse",
        tokenize_pretokenized=True,
        verbose=False,
    )

    examples = []
    pred_upos: list[str] = []
    pred_deprel: list[str] = []
    pred_heads: list[int] = []
    token_count_mismatches = 0

    for sentence in test_sentences:
        examples.extend(sentence_to_examples(sentence))
        token_rows = [[tok.form for tok in sentence.tokens]]
        doc = nlp(token_rows)
        words = doc.sentences[0].words if doc.sentences else []
        if len(words) != len(sentence.tokens):
            token_count_mismatches += 1
        for idx in range(len(sentence.tokens)):
            if idx < len(words):
                word = words[idx]
                pred_upos.append(word.upos or "X")
                pred_deprel.append(word.deprel or "dep")
                pred_heads.append(int(word.head or 0))
            else:
                pred_upos.append("X")
                pred_deprel.append("dep")
                pred_heads.append(0)

    model_scores = evaluate(
        examples=examples,
        predicted_upos=pred_upos,
        predicted_deprel=pred_deprel,
        predicted_heads=pred_heads,
    )
    baseline_scores = evaluate_hybrid_proxy_baseline(examples)

    report = {
        "dataset": {
            "total_sentences": len(all_sentences),
            "test_sentences": len(test_sentences),
            "test_tokens": len(examples),
            "missing_test_ids": missing_ids,
        },
        "split": {
            "seed": split.get("seed", args.seed),
            "source_manifest": str(args.split_manifest),
            "stratified_by_source_sheet": split.get("stratified_by_source_sheet", False),
        },
        "stanza": {
            "lang": args.lang,
            "processors": "tokenize,pos,lemma,depparse",
            "token_count_mismatch_sentences": token_count_mismatches,
        },
        "model_scores": model_scores.__dict__,
        "hybrid_proxy_scores": baseline_scores.__dict__,
        "delta_model_minus_baseline": {
            "upos_accuracy": model_scores.upos_accuracy - baseline_scores.upos_accuracy,
            "deprel_f1_weighted": model_scores.deprel_f1_weighted
            - baseline_scores.deprel_f1_weighted,
            "uas": model_scores.uas - baseline_scores.uas,
            "las": model_scores.las - baseline_scores.las,
        },
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
