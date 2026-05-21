from __future__ import annotations



import argparse

import json

import random

import subprocess

import sys

from pathlib import Path



from kathnlp.evaluation.metrics import evaluate, evaluate_hybrid_proxy_baseline

from kathnlp.training.dataset import load_conllu, sentence_to_examples





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description="Train and evaluate custom Stanza POS+depparse on gold CoNLL-U."

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

        "--output-root",

        type=Path,

        default=Path("src/kathnlp/models/stanza_custom_v1"),

    )

    parser.add_argument(

        "--report-output",

        type=Path,

        default=Path("reports/stanza_custom_v1_report.json"),

    )

    parser.add_argument("--lang", type=str, default="el")

    parser.add_argument("--shorthand", type=str, default="el_kath")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--dev-ratio", type=float, default=0.1)

    parser.add_argument("--max-steps", type=int, default=1200)

    parser.add_argument("--eval-interval", type=int, default=100)

    return parser.parse_args()





def _write_conllu(sentences, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(

        "\n\n".join(sentence.to_conllu() for sentence in sentences) + "\n\n",

        encoding="utf-8",

    )





def _is_sentence_tree_valid(sentence) -> bool:

    tokens = sentence.tokens

    n = len(tokens)

    if n == 0:

        return False

    ids = [tok.id for tok in tokens]

    if ids != list(range(1, n + 1)):

        return False

    head_by_id = {}

    for tok in tokens:

        if tok.head < 0 or tok.head > n:

            return False

        if tok.head == tok.id:

            return False

        head_by_id[tok.id] = tok.head



    color = {i: 0 for i in range(1, n + 1)}



    def visit(node: int) -> bool:

        if color[node] == 1:

            return False

        if color[node] == 2:

            return True

        color[node] = 1

        parent = head_by_id[node]

        if parent != 0 and not visit(parent):

            return False

        color[node] = 2

        return True



    return all(visit(i) for i in range(1, n + 1))





def _to_blind_conllu(sentence) -> str:

    metadata = [f"# sent_id = {sentence.sentence_id}"]

    text = sentence.metadata.get("text")

    if text:

        metadata.append(f"# text = {text}")

    token_lines = []

    for tok in sentence.tokens:

        misc = tok.misc if tok.misc and tok.misc != "_" else "_"

        token_lines.append(

            f"{tok.id}\t{tok.form}\t_\t_\t_\t_\t0\tdep\t_\t{misc}"

        )

    return "\n".join(metadata + token_lines)





def _write_blind_conllu(sentences, path: Path) -> None:

    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(

        "\n\n".join(_to_blind_conllu(sentence) for sentence in sentences) + "\n\n",

        encoding="utf-8",

    )





def _load_split_sentences(

    gold_path: Path, split_manifest: Path, seed: int, dev_ratio: float

) -> tuple[list, list, list, list, dict]:

    all_sentences = load_conllu(gold_path)

    by_id = {s.sentence_id: s for s in all_sentences}

    split = json.loads(split_manifest.read_text(encoding="utf-8"))



    train_pool_raw = [by_id[sid] for sid in split["train_sent_ids"] if sid in by_id]

    test_sentences = [by_id[sid] for sid in split["test_sent_ids"] if sid in by_id]

    train_pool = [s for s in train_pool_raw if _is_sentence_tree_valid(s)]



    rng = random.Random(seed)

    shuffled = train_pool[:]

    rng.shuffle(shuffled)

    dev_size = max(1, int(len(shuffled) * dev_ratio))

    dev_sentences = shuffled[:dev_size]

    train_sentences = shuffled[dev_size:]

    return train_sentences, dev_sentences, test_sentences, train_pool_raw, split





def _run(cmd: list[str]) -> None:

    print("Running:", " ".join(cmd))

    subprocess.run(cmd, check=True)





def main() -> None:

    args = parse_args()



    data_dir = args.output_root / "data"

    data_dir.mkdir(parents=True, exist_ok=True)



    train_sentences, dev_sentences, test_sentences, train_pool_raw, split = _load_split_sentences(

        gold_path=args.gold_path,

        split_manifest=args.split_manifest,

        seed=args.seed,

        dev_ratio=args.dev_ratio,

    )



    train_file = data_dir / f"{args.shorthand}.train.in.conllu"

    dev_file = data_dir / f"{args.shorthand}.dev.in.conllu"

    test_file = data_dir / f"{args.shorthand}.test.in.conllu"

    pos_test_pred_file = data_dir / f"{args.shorthand}.test.pos.pred.conllu"

    final_test_pred_file = data_dir / f"{args.shorthand}.test.pred.conllu"



    _write_conllu(train_sentences, train_file)

    _write_conllu(dev_sentences, dev_file)

    _write_blind_conllu(test_sentences, test_file)



    save_dir = args.output_root

    save_dir.mkdir(parents=True, exist_ok=True)



    pos_model_name = "kath_pos.pt"

    dep_model_name = "kath_depparse.pt"



    common_train_args = [

        "--lang",

        args.lang,

        "--shorthand",

        args.shorthand,

        "--max_steps",

        str(args.max_steps),

        "--eval_interval",

        str(args.eval_interval),

        "--seed",

        str(args.seed),

        "--save_dir",

        str(save_dir),

        "--no_pretrain",

        "--no_char",

        "--no_bert_model",

        "--cpu",

    ]



    _run(

        [

            sys.executable,

            "-m",

            "stanza.models.tagger",

            "--mode",

            "train",

            "--train_file",

            str(train_file),

            "--eval_file",

            str(dev_file),

            "--save_name",

            pos_model_name,

            "--batch_size",

            "64",

            *common_train_args,

        ]

    )



    _run(

        [

            sys.executable,

            "-m",

            "stanza.models.tagger",

            "--mode",

            "predict",

            "--eval_file",

            str(test_file),

            "--output_file",

            str(pos_test_pred_file),

            "--save_name",

            pos_model_name,

            "--no_gold_labels",

            *common_train_args,

        ]

    )



    _run(

        [

            sys.executable,

            "-m",

            "stanza.models.parser",

            "--mode",

            "train",

            "--train_file",

            str(train_file),

            "--eval_file",

            str(dev_file),

            "--save_name",

            dep_model_name,

            "--batch_size",

            "2000",

            *common_train_args,

        ]

    )



    _run(

        [

            sys.executable,

            "-m",

            "stanza.models.parser",

            "--mode",

            "predict",

            "--eval_file",

            str(pos_test_pred_file),

            "--output_file",

            str(final_test_pred_file),

            "--save_name",

            dep_model_name,

            "--no_gold_labels",

            *common_train_args,

        ]

    )



    pred_sentences = load_conllu(final_test_pred_file)

    pred_by_id = {s.sentence_id: s for s in pred_sentences}



    examples = []

    pred_upos: list[str] = []

    pred_deprel: list[str] = []

    pred_heads: list[int] = []

    missing_predictions = 0



    for gold_sentence in test_sentences:

        examples.extend(sentence_to_examples(gold_sentence))

        predicted_sentence = pred_by_id.get(gold_sentence.sentence_id)

        if predicted_sentence is None or len(predicted_sentence.tokens) != len(gold_sentence.tokens):

            missing_predictions += 1

            for _ in gold_sentence.tokens:

                pred_upos.append("X")

                pred_deprel.append("dep")

                pred_heads.append(0)

            continue

        for token in predicted_sentence.tokens:

            pred_upos.append(token.upos)

            pred_deprel.append(token.deprel)

            pred_heads.append(token.head)



    model_scores = evaluate(

        examples=examples,

        predicted_upos=pred_upos,

        predicted_deprel=pred_deprel,

        predicted_heads=pred_heads,

    )

    baseline_scores = evaluate_hybrid_proxy_baseline(examples)



    report = {

        "dataset": {

            "total_sentences": len(load_conllu(args.gold_path)),

            "train_pool_sentences_manifest": len(train_pool_raw),

            "train_pool_sentences_tree_valid": len(train_sentences) + len(dev_sentences),

            "train_sentences": len(train_sentences),

            "dev_sentences": len(dev_sentences),

            "test_sentences": len(test_sentences),

            "test_tokens": len(examples),

        },

        "split": {

            "seed": split.get("seed", args.seed),

            "source_manifest": str(args.split_manifest),

            "custom_dev_ratio_from_train": args.dev_ratio,

            "stratified_by_source_sheet": split.get("stratified_by_source_sheet", False),

        },

        "training": {

            "lang": args.lang,

            "shorthand": args.shorthand,

            "max_steps": args.max_steps,

            "eval_interval": args.eval_interval,

            "save_dir": str(save_dir),

            "pos_model_name": pos_model_name,

            "dep_model_name": dep_model_name,

        },

        "artifacts": {

            "train_file": str(train_file),

            "dev_file": str(dev_file),

            "test_file": str(test_file),

            "pos_test_pred_file": str(pos_test_pred_file),

            "final_test_pred_file": str(final_test_pred_file),

            "missing_or_misaligned_sentences": missing_predictions,

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

