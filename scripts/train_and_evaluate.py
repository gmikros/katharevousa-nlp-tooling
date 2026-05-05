from __future__ import annotations



import argparse

import json

from collections import Counter, defaultdict

from pathlib import Path



from sklearn.model_selection import train_test_split



from kathnlp.evaluation.metrics import evaluate, evaluate_hybrid_proxy_baseline

from kathnlp.training.dataset import load_conllu, sentences_to_examples

from kathnlp.training.models import train_token_tagger





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description="Train and evaluate Katharevousa pilot models.")

    parser.add_argument(

        "--gold-path",

        type=Path,

        default=Path("data/processed/pilot/gold_provisional.conllu"),

    )

    parser.add_argument(

        "--model-output",

        type=Path,

        default=Path("src/kathnlp/models/token_tagger.joblib"),

    )

    parser.add_argument(

        "--report-output",

        type=Path,

        default=Path("reports/pilot_eval_report.json"),

    )

    parser.add_argument(

        "--split-manifest-output",

        type=Path,

        default=Path("reports/train_split_manifest.json"),

    )

    parser.add_argument(

        "--error-report-output",

        type=Path,

        default=Path("reports/parser_error_report.json"),

    )

    parser.add_argument(

        "--hardcase-error-report",

        type=Path,

        default=None,

        help="Optional parser error report used to prioritize deprel confusions.",

    )

    parser.add_argument(

        "--hardcase-multiplier",

        type=int,

        default=1,

        help="How many times to include selected hard-case examples in train set.",

    )

    parser.add_argument(

        "--hardcase-min-sentence-len",

        type=int,

        default=21,

        help="Minimum sentence length considered difficult for upweighting.",

    )

    parser.add_argument(

        "--hardcase-min-head-distance",

        type=int,

        default=6,

        help="Minimum absolute head distance considered difficult for upweighting.",

    )

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()





def _length_bucket(sentence_len: int) -> str:

    if sentence_len <= 10:

        return "01_10"

    if sentence_len <= 20:

        return "11_20"

    if sentence_len <= 30:

        return "21_30"

    return "31_plus"





def _head_distance_bucket(head: int, token_id: int) -> str:

    if head == 0:

        return "root"

    dist = abs(head - token_id)

    if dist <= 2:

        return "1_2"

    if dist <= 5:

        return "3_5"

    if dist <= 10:

        return "6_10"

    return "11_plus"





def _build_error_report(

    test_examples, pred_upos: list[str], pred_deprel: list[str], pred_heads: list[int]

) -> dict:

    las_bucket_correct = Counter()

    las_bucket_total = Counter()

    uas_bucket_correct = Counter()

    uas_bucket_total = Counter()

    deprel_confusions = Counter()

    sentence_errors = defaultdict(int)



    for ex, pu, pd, ph in zip(test_examples, pred_upos, pred_deprel, pred_heads):

        len_bucket = _length_bucket(ex.sentence_len)

        las_bucket_total[len_bucket] += 1

        uas_bucket_total[len_bucket] += 1

        uas_ok = ex.head == ph

        las_ok = uas_ok and ex.deprel == pd

        if uas_ok:

            uas_bucket_correct[len_bucket] += 1

        if las_ok:

            las_bucket_correct[len_bucket] += 1

        if not las_ok:

            sentence_errors[ex.sent_id] += 1

        if ex.deprel != pd:

            deprel_confusions[(ex.deprel, pd)] += 1



    head_bucket_correct = Counter()

    head_bucket_total = Counter()

    for ex, ph in zip(test_examples, pred_heads):

        b = _head_distance_bucket(ex.head, ex.token_id)

        head_bucket_total[b] += 1

        if ex.head == ph:

            head_bucket_correct[b] += 1



    return {

        "token_count": len(test_examples),

        "las_by_sentence_length": {

            k: las_bucket_correct[k] / max(1, las_bucket_total[k])

            for k in sorted(las_bucket_total)

        },

        "uas_by_sentence_length": {

            k: uas_bucket_correct[k] / max(1, uas_bucket_total[k])

            for k in sorted(uas_bucket_total)

        },

        "uas_by_gold_head_distance": {

            k: head_bucket_correct[k] / max(1, head_bucket_total[k])

            for k in sorted(head_bucket_total)

        },

        "top_deprel_confusions": [

            {"gold": gold, "pred": pred, "count": count}

            for (gold, pred), count in deprel_confusions.most_common(20)

        ],

        "top_sentences_by_las_errors": [

            {"sent_id": sent_id, "las_errors": count}

            for sent_id, count in sorted(sentence_errors.items(), key=lambda x: x[1], reverse=True)[:20]

        ],

    }





def _load_target_deprels(error_report_path: Path | None) -> set[str]:

    if error_report_path is None or not error_report_path.exists():

        return set()

    payload = json.loads(error_report_path.read_text(encoding="utf-8"))

    target: set[str] = set()

    for row in payload.get("top_deprel_confusions", []):

        gold = row.get("gold")

        pred = row.get("pred")

        if isinstance(gold, str):

            target.add(gold)

        if isinstance(pred, str):

            target.add(pred)

    return target





def _augment_hard_cases(

    train_examples,

    target_deprels: set[str],

    multiplier: int,

    min_sentence_len: int,

    min_head_distance: int,

) -> tuple[list, dict[str, int]]:

    if multiplier <= 1:

        return train_examples, {

            "hardcase_examples": 0,

            "hardcase_repeats_added": 0,

            "augmented_train_examples": len(train_examples),

        }

    hard_examples = []

    for ex in train_examples:

        long_sentence = ex.sentence_len >= min_sentence_len

        head_distance = ex.head != 0 and abs(ex.head - ex.token_id) >= min_head_distance

        deprel_confused = ex.deprel in target_deprels if target_deprels else False

        if (long_sentence and (head_distance or deprel_confused)) or (

            head_distance and deprel_confused

        ):

            hard_examples.append(ex)

    repeats = hard_examples * (multiplier - 1)

    augmented = train_examples + repeats

    return augmented, {

        "hardcase_examples": len(hard_examples),

        "hardcase_repeats_added": len(repeats),

        "augmented_train_examples": len(augmented),

    }





def main() -> None:

    args = parse_args()

    sentences = load_conllu(args.gold_path)

    stratify_labels = [s.metadata.get("source_sheet", "unknown") for s in sentences]

    # Stratify only when it is statistically valid.

    can_stratify = len(set(stratify_labels)) > 1 and min(

        stratify_labels.count(label) for label in set(stratify_labels)

    ) > 1

    train_sentences, test_sentences = train_test_split(

        sentences,

        test_size=0.2,

        random_state=args.seed,

        stratify=stratify_labels if can_stratify else None,

    )

    train_examples = sentences_to_examples(train_sentences)

    target_deprels = _load_target_deprels(args.hardcase_error_report)

    train_examples_aug, hardcase_stats = _augment_hard_cases(

        train_examples=train_examples,

        target_deprels=target_deprels,

        multiplier=args.hardcase_multiplier,

        min_sentence_len=args.hardcase_min_sentence_len,

        min_head_distance=args.hardcase_min_head_distance,

    )

    test_examples = sentences_to_examples(test_sentences)



    model = train_token_tagger(train_examples_aug)

    args.model_output.parent.mkdir(parents=True, exist_ok=True)

    model.save(str(args.model_output))



    pred_upos, pred_deprel, pred_heads = model.predict(test_examples)

    model_scores = evaluate(test_examples, pred_upos, pred_deprel, predicted_heads=pred_heads)

    baseline_scores = evaluate_hybrid_proxy_baseline(test_examples)



    report = {

        "dataset": {

            "total_sentences": len(sentences),

            "train_sentences": len(train_sentences),

            "test_sentences": len(test_sentences),

            "train_tokens": len(train_examples),

            "train_tokens_after_hardcase_aug": len(train_examples_aug),

            "test_tokens": len(test_examples),

        },

        "hardcase_augmentation": {

            "enabled": args.hardcase_multiplier > 1,

            "error_report_source": str(args.hardcase_error_report)

            if args.hardcase_error_report

            else None,

            "target_deprel_count": len(target_deprels),

            "target_deprels": sorted(target_deprels),

            "hardcase_multiplier": args.hardcase_multiplier,

            "min_sentence_len": args.hardcase_min_sentence_len,

            "min_head_distance": args.hardcase_min_head_distance,

            **hardcase_stats,

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

    split_manifest = {

        "seed": args.seed,

        "stratified_by_source_sheet": can_stratify,

        "train_sent_ids": [s.sentence_id for s in train_sentences],

        "test_sent_ids": [s.sentence_id for s in test_sentences],

    }

    args.split_manifest_output.parent.mkdir(parents=True, exist_ok=True)

    args.split_manifest_output.write_text(

        json.dumps(split_manifest, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    error_report = _build_error_report(test_examples, pred_upos, pred_deprel, pred_heads)

    args.error_report_output.parent.mkdir(parents=True, exist_ok=True)

    args.error_report_output.write_text(

        json.dumps(error_report, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    args.report_output.parent.mkdir(parents=True, exist_ok=True)

    args.report_output.write_text(

        json.dumps(report, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    print(json.dumps(report, ensure_ascii=False, indent=2))





if __name__ == "__main__":

    main()

