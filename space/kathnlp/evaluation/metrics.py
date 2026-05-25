from __future__ import annotations



from dataclasses import dataclass



from sklearn.metrics import accuracy_score, f1_score



from kathnlp.training.dataset import TokenExample





@dataclass

class EvalScores:

    upos_accuracy: float

    deprel_f1_weighted: float

    uas: float

    las: float





def _predict_heads_heuristic(examples: list[TokenExample], predicted_upos: list[str]) -> list[int]:

    # Simple dependency proxy:

    # first predicted VERB becomes root, otherwise token 1 is root.

    root_id = 1

    for ex, upos in zip(examples, predicted_upos):

        if upos == "VERB":

            root_id = ex.token_id

            break

    predicted = []

    for ex in examples:

        if ex.token_id == root_id:

            predicted.append(0)

        else:

            predicted.append(root_id)

    return predicted





def evaluate(

    examples: list[TokenExample],

    predicted_upos: list[str],

    predicted_deprel: list[str],

    predicted_heads: list[int] | None = None,

) -> EvalScores:

    gold_upos = [ex.upos for ex in examples]

    gold_deprel = [ex.deprel for ex in examples]

    gold_head = [ex.head for ex in examples]



    upos_acc = accuracy_score(gold_upos, predicted_upos)

    deprel_f1 = f1_score(gold_deprel, predicted_deprel, average="weighted")



    pred_head = predicted_heads or _predict_heads_heuristic(examples, predicted_upos)

    uas = sum(int(a == b) for a, b in zip(gold_head, pred_head)) / max(1, len(gold_head))

    las = (

        sum(

            int((gh == ph) and (gd == pd))

            for gh, ph, gd, pd in zip(gold_head, pred_head, gold_deprel, predicted_deprel)

        )

        / max(1, len(gold_head))

    )



    return EvalScores(

        upos_accuracy=upos_acc,

        deprel_f1_weighted=deprel_f1,

        uas=uas,

        las=las,

    )





def evaluate_hybrid_proxy_baseline(examples: list[TokenExample]) -> EvalScores:

    # Proxy baseline approximating weak out-of-domain behavior.

    predicted_upos = ["NOUN" if ex.form.isalpha() else "PUNCT" for ex in examples]

    predicted_deprel = ["dep" for _ in examples]

    return evaluate(examples, predicted_upos, predicted_deprel)

