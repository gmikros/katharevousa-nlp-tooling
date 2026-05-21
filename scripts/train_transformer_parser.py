from __future__ import annotations



import argparse

import json

from pathlib import Path



from sklearn.model_selection import train_test_split



from kathnlp.training.dataset import load_conllu

from kathnlp.training.transformer_parser import (

    TrainConfig,

    build_label_maps,

    evaluate_transformer_parser,

    save_transformer_parser,

    train_transformer_parser,

)





def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(

        description="Train a transformer-based dependency parser for Katharevousa."

    )

    parser.add_argument(

        "--gold-path",

        type=Path,

        default=Path("data/processed/final_gold/gold_final.conllu"),

    )

    parser.add_argument(

        "--output-dir",

        type=Path,

        default=Path("src/kathnlp/models/transformer_parser_v1"),

    )

    parser.add_argument(

        "--report-output",

        type=Path,

        default=Path("reports/transformer_parser_eval_report.json"),

    )

    parser.add_argument(

        "--split-manifest-output",

        type=Path,

        default=Path("reports/transformer_parser_split_manifest.json"),

    )

    parser.add_argument("--encoder-name", type=str, default="xlm-roberta-base")

    parser.add_argument("--epochs", type=int, default=6)

    parser.add_argument("--batch-size", type=int, default=8)

    parser.add_argument("--learning-rate", type=float, default=2e-5)

    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--max-length", type=int, default=256)

    parser.add_argument("--warmup-ratio", type=float, default=0.1)

    parser.add_argument("--upos-weight", type=float, default=1.0)

    parser.add_argument("--arc-weight", type=float, default=1.8)

    parser.add_argument("--rel-weight", type=float, default=1.2)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()





def main() -> None:

    args = parse_args()

    sentences = load_conllu(args.gold_path)

    stratify_labels = [s.metadata.get("source_sheet", "unknown") for s in sentences]

    can_stratify = len(set(stratify_labels)) > 1 and min(

        stratify_labels.count(label) for label in set(stratify_labels)

    ) > 1

    train_sentences, test_sentences = train_test_split(

        sentences,

        test_size=0.2,

        random_state=args.seed,

        stratify=stratify_labels if can_stratify else None,

    )

    labels = build_label_maps(train_sentences)

    config = TrainConfig(

        encoder_name=args.encoder_name,

        epochs=args.epochs,

        batch_size=args.batch_size,

        learning_rate=args.learning_rate,

        weight_decay=args.weight_decay,

        max_length=args.max_length,

        warmup_ratio=args.warmup_ratio,

        upos_weight=args.upos_weight,

        arc_weight=args.arc_weight,

        rel_weight=args.rel_weight,

    )

    model, tokenizer, history = train_transformer_parser(

        train_sentences=train_sentences,

        label_maps=labels,

        config=config,

        seed=args.seed,

    )

    model_scores, baseline_scores = evaluate_transformer_parser(

        model=model,

        tokenizer=tokenizer,

        test_sentences=test_sentences,

        label_maps=labels,

        max_length=args.max_length,

    )

    save_transformer_parser(

        model=model,

        tokenizer=tokenizer,

        label_maps=labels,

        output_dir=args.output_dir,

        train_config=config,

        train_history=history,

    )

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

    report = {

        "dataset": {

            "total_sentences": len(sentences),

            "train_sentences": len(train_sentences),

            "test_sentences": len(test_sentences),

        },

        "training": {

            "encoder_name": args.encoder_name,

            "epochs": args.epochs,

            "batch_size": args.batch_size,

            "learning_rate": args.learning_rate,

            "weight_decay": args.weight_decay,

            "max_length": args.max_length,

            "warmup_ratio": args.warmup_ratio,

            "upos_weight": args.upos_weight,

            "arc_weight": args.arc_weight,

            "rel_weight": args.rel_weight,

            "history": history,

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

        "output_dir": str(args.output_dir),

    }

    args.report_output.parent.mkdir(parents=True, exist_ok=True)

    args.report_output.write_text(

        json.dumps(report, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    print(json.dumps(report, ensure_ascii=False, indent=2))





if __name__ == "__main__":

    main()

