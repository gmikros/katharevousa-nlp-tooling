from __future__ import annotations







import json



import random



from dataclasses import dataclass



from pathlib import Path







import numpy as np



import torch



import torch.nn.functional as F



from torch import nn



from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup







from kathnlp.evaluation.metrics import EvalScores, evaluate, evaluate_hybrid_proxy_baseline



from kathnlp.schema import UD_DEPREL, UD_UPOS, UDSentenceAnnotation



from kathnlp.training.dataset import sentence_to_examples











def set_seed(seed: int) -> None:



    random.seed(seed)



    np.random.seed(seed)



    torch.manual_seed(seed)



    if torch.cuda.is_available():



        torch.cuda.manual_seed_all(seed)











@dataclass



class ParserLabelMaps:



    upos_to_id: dict[str, int]



    upos_labels: list[str]



    deprel_to_id: dict[str, int]



    deprel_labels: list[str]











def build_label_maps(sentences: list[UDSentenceAnnotation]) -> ParserLabelMaps:



    upos_values = sorted({tok.upos for s in sentences for tok in s.tokens} | set(UD_UPOS))



    deprel_values = sorted(



        {tok.deprel for s in sentences for tok in s.tokens} | set(UD_DEPREL)



    )



    upos_to_id = {label: i for i, label in enumerate(upos_values)}



    deprel_to_id = {label: i for i, label in enumerate(deprel_values)}



    return ParserLabelMaps(



        upos_to_id=upos_to_id,



        upos_labels=upos_values,



        deprel_to_id=deprel_to_id,



        deprel_labels=deprel_values,



    )











class TransformerDependencyParser(nn.Module):



    def __init__(



        self,



        encoder_name: str,



        upos_count: int,



        deprel_count: int,



        arc_dim: int = 384,



        dropout: float = 0.2,



    ) -> None:



        super().__init__()



        self.encoder = AutoModel.from_pretrained(encoder_name)



        hidden = self.encoder.config.hidden_size



        self.dropout = nn.Dropout(dropout)



        self.upos_classifier = nn.Linear(hidden, upos_count)



        self.dep_arc = nn.Sequential(nn.Linear(hidden, arc_dim), nn.ReLU(), nn.Dropout(dropout))



        self.head_arc = nn.Sequential(nn.Linear(hidden, arc_dim), nn.ReLU(), nn.Dropout(dropout))



        self.root_arc = nn.Parameter(torch.randn(arc_dim))



        self.root_rel = nn.Parameter(torch.randn(hidden))



        self.rel_classifier = nn.Sequential(



            nn.Linear(hidden * 2, hidden),



            nn.ReLU(),



            nn.Dropout(dropout),



            nn.Linear(hidden, deprel_count),



        )







    def _sentence_scores(



        self, word_reps: torch.Tensor, head_targets: torch.Tensor | None



    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:



        dep_proj = self.dep_arc(self.dropout(word_reps))



        head_proj = self.head_arc(self.dropout(word_reps))



        head_candidates = torch.cat(



            [self.root_arc.unsqueeze(0), head_proj],



            dim=0,



        )



        arc_scores = dep_proj @ head_candidates.T



        if head_targets is None:



            chosen_heads = arc_scores.argmax(dim=-1)



        else:



            chosen_heads = head_targets







        rel_candidates = torch.cat([self.root_rel.unsqueeze(0), word_reps], dim=0)



        chosen_head_reps = rel_candidates[chosen_heads]



        rel_inputs = torch.cat([word_reps, chosen_head_reps], dim=-1)



        rel_logits = self.rel_classifier(self.dropout(rel_inputs))



        upos_logits = self.upos_classifier(self.dropout(word_reps))



        return upos_logits, arc_scores, rel_logits







    def training_loss(



        self,



        word_reps_list: list[torch.Tensor],



        upos_targets: list[torch.Tensor],



        head_targets: list[torch.Tensor],



        deprel_targets: list[torch.Tensor],



        upos_weight: float = 1.0,



        arc_weight: float = 1.0,



        rel_weight: float = 1.0,



    ) -> torch.Tensor:



        total = torch.tensor(0.0, device=word_reps_list[0].device)



        token_count = 0



        for reps, upos_y, head_y, dep_y in zip(



            word_reps_list, upos_targets, head_targets, deprel_targets



        ):



            upos_logits, arc_scores, rel_logits = self._sentence_scores(reps, head_y)



            total = (



                total



                + upos_weight * F.cross_entropy(upos_logits, upos_y)



                + arc_weight * F.cross_entropy(arc_scores, head_y)



                + rel_weight * F.cross_entropy(rel_logits, dep_y)



            )



            token_count += reps.size(0)



        return total / max(1, token_count)







    @torch.no_grad()



    def predict_sentence(



        self, word_reps: torch.Tensor



    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:



        upos_logits, arc_scores, rel_logits = self._sentence_scores(word_reps, head_targets=None)



        return (



            upos_logits.argmax(dim=-1),



            arc_scores.argmax(dim=-1),



            rel_logits.argmax(dim=-1),



        )











def _batch_to_word_reps(



    tokenizer,



    model: TransformerDependencyParser,



    token_batches: list[list[str]],



    device: torch.device,



    max_length: int,



) -> tuple[list[torch.Tensor], list[list[int]]]:



    encoded_batch = tokenizer(



        token_batches,



        is_split_into_words=True,



        return_tensors="pt",



        padding=True,



        truncation=True,



        max_length=max_length,



    )



    model_inputs = {k: v.to(device) for k, v in encoded_batch.items()}



    hidden = model.encoder(**model_inputs).last_hidden_state







    word_reps_list: list[torch.Tensor] = []



    kept_word_ids_list: list[list[int]] = []



    for batch_idx, tokens in enumerate(token_batches):



        word_ids = encoded_batch.word_ids(batch_index=batch_idx)



        first_positions: list[int] = []



        seen: set[int] = set()



        kept_word_ids: list[int] = []



        for pos, wid in enumerate(word_ids):



            if wid is None or wid in seen:



                continue



            seen.add(wid)



            first_positions.append(pos)



            kept_word_ids.append(wid)



        if not first_positions:



            first_positions = [0]



            kept_word_ids = [0]



        idx_tensor = torch.tensor(first_positions, dtype=torch.long, device=device)



        word_reps_list.append(hidden[batch_idx].index_select(0, idx_tensor))



        kept_word_ids_list.append(kept_word_ids)



    return word_reps_list, kept_word_ids_list











def _sentence_targets(



    sentence: UDSentenceAnnotation,



    labels: ParserLabelMaps,



    device: torch.device,



    kept_word_ids: list[int] | None = None,



) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:



    if kept_word_ids is None:



        kept_word_ids = list(range(len(sentence.tokens)))



    old_to_new = {old_idx + 1: new_idx + 1 for new_idx, old_idx in enumerate(kept_word_ids)}



    kept_tokens = [sentence.tokens[i] for i in kept_word_ids]



    upos = torch.tensor(



        [labels.upos_to_id[tok.upos] for tok in kept_tokens],



        dtype=torch.long,



        device=device,



    )



    heads = torch.tensor(



        [old_to_new.get(tok.head, 0) for tok in kept_tokens],



        dtype=torch.long,



        device=device,



    )



    deprel = torch.tensor(



        [labels.deprel_to_id[tok.deprel] for tok in kept_tokens],



        dtype=torch.long,



        device=device,



    )



    return upos, heads, deprel











@dataclass



class TrainConfig:



    encoder_name: str = "xlm-roberta-base"



    epochs: int = 6



    batch_size: int = 8



    learning_rate: float = 2e-5



    weight_decay: float = 0.01



    max_length: int = 256



    upos_weight: float = 1.0



    arc_weight: float = 1.8



    rel_weight: float = 1.2



    warmup_ratio: float = 0.1











def train_transformer_parser(



    train_sentences: list[UDSentenceAnnotation],



    label_maps: ParserLabelMaps,



    config: TrainConfig,



    seed: int = 42,



) -> tuple[TransformerDependencyParser, object, list[dict[str, float]]]:

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name, use_fast=True)

    model = TransformerDependencyParser(

        encoder_name=config.encoder_name,

        upos_count=len(label_maps.upos_labels),

        deprel_count=len(label_maps.deprel_labels),

    ).to(device)



    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.learning_rate,

        weight_decay=config.weight_decay,

    )

    steps_per_epoch = max(1, (len(train_sentences) + config.batch_size - 1) // config.batch_size)

    total_steps = steps_per_epoch * config.epochs

    warmup_steps = int(total_steps * config.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(

        optimizer=optimizer,

        num_warmup_steps=warmup_steps,

        num_training_steps=total_steps,

    )



    history: list[dict[str, float]] = []

    global_step = 0

    for epoch in range(1, config.epochs + 1):

        model.train()

        shuffled = train_sentences[:]

        random.shuffle(shuffled)

        batch_losses: list[float] = []

        for start in range(0, len(shuffled), config.batch_size):

            batch = shuffled[start : start + config.batch_size]

            tokens = [[tok.form for tok in sentence.tokens] for sentence in batch]

            word_reps_list, kept_word_ids_list = _batch_to_word_reps(

                tokenizer=tokenizer,

                model=model,

                token_batches=tokens,

                device=device,

                max_length=config.max_length,

            )

            upos_y: list[torch.Tensor] = []

            head_y: list[torch.Tensor] = []

            dep_y: list[torch.Tensor] = []

            for sentence, kept_word_ids in zip(batch, kept_word_ids_list):

                u, h, d = _sentence_targets(

                    sentence,

                    labels=label_maps,

                    device=device,

                    kept_word_ids=kept_word_ids,

                )

                upos_y.append(u)

                head_y.append(h)

                dep_y.append(d)



            loss = model.training_loss(

                word_reps_list=word_reps_list,

                upos_targets=upos_y,

                head_targets=head_y,

                deprel_targets=dep_y,

                upos_weight=config.upos_weight,

                arc_weight=config.arc_weight,

                rel_weight=config.rel_weight,

            )

            optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            scheduler.step()

            global_step += 1

            batch_losses.append(loss.item())



        history.append(

            {

                "epoch": float(epoch),

                "loss": float(np.mean(batch_losses)),

                "learning_rate": float(scheduler.get_last_lr()[0]),

                "global_step": float(global_step),

            }

        )

    return model, tokenizer, history











@torch.no_grad()



def predict_sentences(



    model: TransformerDependencyParser,



    tokenizer,



    sentences: list[UDSentenceAnnotation],



    label_maps: ParserLabelMaps,



    max_length: int = 256,



) -> tuple[list[str], list[str], list[int]]:



    device = next(model.parameters()).device



    model.eval()



    pred_upos: list[str] = []



    pred_deprel: list[str] = []



    pred_heads: list[int] = []



    for sentence in sentences:



        token_batches = [[tok.form for tok in sentence.tokens]]



        word_reps, kept_word_ids_list = _batch_to_word_reps(



            tokenizer=tokenizer,



            model=model,



            token_batches=token_batches,



            device=device,



            max_length=max_length,



        )



        sentence_word_reps = word_reps[0]



        kept_word_ids = kept_word_ids_list[0]



        upos_ids, head_ids, dep_ids = model.predict_sentence(sentence_word_reps)



        old_to_new = {old_idx + 1: new_idx + 1 for new_idx, old_idx in enumerate(kept_word_ids)}



        new_to_old = {v: k for k, v in old_to_new.items()}



        predicted_by_old: dict[int, tuple[str, str, int]] = {}



        for new_idx, old_idx in enumerate(kept_word_ids, start=1):



            pred_head_new = int(head_ids[new_idx - 1].item())



            pred_head_old = new_to_old.get(pred_head_new, 0)



            predicted_by_old[old_idx + 1] = (



                label_maps.upos_labels[int(upos_ids[new_idx - 1].item())],



                label_maps.deprel_labels[int(dep_ids[new_idx - 1].item())],



                pred_head_old,



            )



        for token in sentence.tokens:



            pred = predicted_by_old.get(token.id)



            if pred is None:



                pred_upos.append("X")



                pred_deprel.append("dep")



                pred_heads.append(0)



            else:



                pred_upos.append(pred[0])



                pred_deprel.append(pred[1])



                pred_heads.append(pred[2])



    return pred_upos, pred_deprel, pred_heads











def evaluate_transformer_parser(



    model: TransformerDependencyParser,



    tokenizer,



    test_sentences: list[UDSentenceAnnotation],



    label_maps: ParserLabelMaps,



    max_length: int = 256,



) -> tuple[EvalScores, EvalScores]:



    pred_upos, pred_deprel, pred_heads = predict_sentences(



        model=model,



        tokenizer=tokenizer,



        sentences=test_sentences,



        label_maps=label_maps,



        max_length=max_length,



    )



    examples = [ex for sentence in test_sentences for ex in sentence_to_examples(sentence)]



    model_scores = evaluate(



        examples=examples,



        predicted_upos=pred_upos,



        predicted_deprel=pred_deprel,



        predicted_heads=pred_heads,



    )



    baseline_scores = evaluate_hybrid_proxy_baseline(examples)



    return model_scores, baseline_scores











def save_transformer_parser(



    model: TransformerDependencyParser,



    tokenizer,



    label_maps: ParserLabelMaps,



    output_dir: Path,



    train_config: TrainConfig,



    train_history: list[dict[str, float]],



) -> None:



    output_dir.mkdir(parents=True, exist_ok=True)



    encoder_dir = output_dir / "encoder"



    tokenizer_dir = output_dir / "tokenizer"



    model.encoder.save_pretrained(encoder_dir)



    tokenizer.save_pretrained(tokenizer_dir)



    torch.save(model.state_dict(), output_dir / "parser_heads.pt")



    metadata = {



        "upos_labels": label_maps.upos_labels,



        "deprel_labels": label_maps.deprel_labels,



        "train_config": train_config.__dict__,



        "train_history": train_history,



    }



    (output_dir / "metadata.json").write_text(



        json.dumps(metadata, ensure_ascii=False, indent=2),



        encoding="utf-8",



    )



