from __future__ import annotations

import argparse
import json
from pathlib import Path

from kathnlp.evaluation.metrics import evaluate
from kathnlp.training.dataset import load_conllu, sentence_to_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate external baselines (spaCy/Stanza) on fixed Katharevousa split."
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
        default=Path("reports/external_baselines_report.json"),
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="el_core_news_lg",
    )
    parser.add_argument(
        "--reference-reports",
        type=Path,
        nargs="*",
        default=[
            Path("reports/transformer_parser_v1_opt_report.json"),
            Path("reports/transformer_parser_mbert_v2_report.json"),
            Path("reports/stanza_custom_v1_report.json"),
            Path("reports/final_gold_eval_report_v3.json"),
        ],
    )
    return parser.parse_args()


def _load_test_sentences(gold_path: Path, split_manifest: Path):
    all_sentences = load_conllu(gold_path)
    by_id = {s.sentence_id: s for s in all_sentences}
    split = json.loads(split_manifest.read_text(encoding="utf-8"))
    test_ids: list[str] = split["test_sent_ids"]
    test_sentences = [by_id[sid] for sid in test_ids if sid in by_id]
    missing = [sid for sid in test_ids if sid not in by_id]
    return all_sentences, test_sentences, split, missing


def _evaluate_spacy(sentences, model_name: str):
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(model_name)
    examples = []
    pred_upos: list[str] = []
    pred_deprel: list[str] = []
    pred_heads: list[int] = []
    mismatch_sentences = 0

    for sentence in sentences:
        tokens = [tok.form for tok in sentence.tokens]
        spaces = [True] * len(tokens)
        if spaces:
            spaces[-1] = False
        doc = Doc(nlp.vocab, words=tokens, spaces=spaces)
        doc = nlp(doc)

        examples.extend(sentence_to_examples(sentence))
        if len(doc) != len(sentence.tokens):
            mismatch_sentences += 1

        for idx in range(len(sentence.tokens)):
            if idx >= len(doc):
                pred_upos.append("X")
                pred_deprel.append("dep")
                pred_heads.append(0)
                continue
            tok = doc[idx]
            upos = tok.pos_ if tok.pos_ else "X"
            deprel = tok.dep_.lower() if tok.dep_ else "dep"
            if deprel == "root" or tok.head.i == tok.i:
                head = 0
            else:
                head = tok.head.i + 1
            pred_upos.append(upos)
            pred_deprel.append(deprel)
            pred_heads.append(head)

    scores = evaluate(
        examples=examples,
        predicted_upos=pred_upos,
        predicted_deprel=pred_deprel,
        predicted_heads=pred_heads,
    )
    return scores, mismatch_sentences


def _evaluate_stanza(sentences, lang: str, package: str | None, label: str):
    import stanza

    download_kwargs = {"lang": lang, "verbose": False}
    if package:
        download_kwargs["package"] = package
    stanza.download(**download_kwargs)

    pipeline_kwargs = {
        "lang": lang,
        "processors": "tokenize,pos,lemma,depparse",
        "tokenize_pretokenized": True,
        "verbose": False,
    }
    if package:
        pipeline_kwargs["package"] = package
    nlp = stanza.Pipeline(**pipeline_kwargs)

    examples = []
    pred_upos: list[str] = []
    pred_deprel: list[str] = []
    pred_heads: list[int] = []
    mismatch_sentences = 0

    for sentence in sentences:
        examples.extend(sentence_to_examples(sentence))
        doc = nlp([[tok.form for tok in sentence.tokens]])
        words = doc.sentences[0].words if doc.sentences else []
        if len(words) != len(sentence.tokens):
            mismatch_sentences += 1
        for idx in range(len(sentence.tokens)):
            if idx >= len(words):
                pred_upos.append("X")
                pred_deprel.append("dep")
                pred_heads.append(0)
                continue
            word = words[idx]
            pred_upos.append(word.upos or "X")
            pred_deprel.append((word.deprel or "dep").lower())
            pred_heads.append(int(word.head or 0))

    scores = evaluate(
        examples=examples,
        predicted_upos=pred_upos,
        predicted_deprel=pred_deprel,
        predicted_heads=pred_heads,
    )
    return scores, mismatch_sentences, label


def _load_reference_scores(paths: list[Path]) -> dict[str, dict[str, float]]:
    refs: dict[str, dict[str, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores = payload.get("model_scores")
        if isinstance(scores, dict):
            refs[path.stem] = scores
    return refs


def main() -> None:
    args = parse_args()
    all_sentences, test_sentences, split, missing = _load_test_sentences(
        args.gold_path, args.split_manifest
    )
    test_examples = [ex for s in test_sentences for ex in sentence_to_examples(s)]

    spacy_scores, spacy_mismatch = _evaluate_spacy(test_sentences, args.spacy_model)
    stanza_el_scores, stanza_el_mismatch, _ = _evaluate_stanza(
        test_sentences, lang="el", package=None, label="stanza_el_pretrained"
    )
    stanza_grc_scores, stanza_grc_mismatch, _ = _evaluate_stanza(
        test_sentences, lang="grc", package="proiel", label="stanza_grc_proiel"
    )

    references = _load_reference_scores(args.reference_reports)

    models = {
        f"spacy_{args.spacy_model}": {
            "scores": spacy_scores.__dict__,
            "token_count_mismatch_sentences": spacy_mismatch,
        },
        "stanza_el_pretrained": {
            "scores": stanza_el_scores.__dict__,
            "token_count_mismatch_sentences": stanza_el_mismatch,
        },
        "stanza_grc_proiel": {
            "scores": stanza_grc_scores.__dict__,
            "token_count_mismatch_sentences": stanza_grc_mismatch,
        },
    }

    ranking = sorted(
        ((name, data["scores"]["las"]) for name, data in models.items()),
        key=lambda x: x[1],
        reverse=True,
    )

    report = {
        "dataset": {
            "total_sentences": len(all_sentences),
            "test_sentences": len(test_sentences),
            "test_tokens": len(test_examples),
            "missing_test_ids": missing,
        },
        "split": {
            "seed": split.get("seed"),
            "source_manifest": str(args.split_manifest),
            "stratified_by_source_sheet": split.get("stratified_by_source_sheet", False),
        },
        "external_models": models,
        "ranking_by_las": [{"model": name, "las": las} for name, las in ranking],
        "reference_models": references,
    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
