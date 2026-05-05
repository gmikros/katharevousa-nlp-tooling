from __future__ import annotations



from collections import Counter

from dataclasses import dataclass

from typing import Any



from kathnlp.schema import UDSentenceAnnotation, UDTokenAnnotation





@dataclass

class VoteResult:

    auto_accepted: UDSentenceAnnotation | None

    disagreement_payload: dict[str, Any] | None





def _majority_value(values: list[Any]) -> tuple[Any | None, bool]:

    counter = Counter(values)

    if not counter:

        return None, False

    value, freq = counter.most_common(1)[0]

    return (value, freq >= 2)





def _token_key(token: UDTokenAnnotation) -> tuple[int, str]:

    return (token.id, token.form)





def vote_on_three_annotations(

    ann_a: UDSentenceAnnotation,

    ann_b: UDSentenceAnnotation,

    ann_c: UDSentenceAnnotation,

) -> VoteResult:

    by_model = [ann_a, ann_b, ann_c]

    base = ann_a



    if not (len(ann_a.tokens) == len(ann_b.tokens) == len(ann_c.tokens)):

        return VoteResult(

            auto_accepted=None,

            disagreement_payload={

                "sent_id": base.sentence_id,

                "text": base.text,

                "reason": "token_length_mismatch",

                "token_counts": [len(ann_a.tokens), len(ann_b.tokens), len(ann_c.tokens)],

                "candidates": [

                    {"model": a.metadata.get("annotator"), "conllu": a.to_conllu()}

                    for a in by_model

                ],

            },

        )



    voted_tokens: list[UDTokenAnnotation] = []

    disagreements: list[dict[str, Any]] = []



    for idx in range(len(base.tokens)):

        tok_a = ann_a.tokens[idx]

        tok_b = ann_b.tokens[idx]

        tok_c = ann_c.tokens[idx]

        if _token_key(tok_a) != _token_key(tok_b) or _token_key(tok_a) != _token_key(tok_c):

            disagreements.append(

                {

                    "token_index": idx,

                    "reason": "token_identity_mismatch",

                    "forms": [tok_a.form, tok_b.form, tok_c.form],

                }

            )

            continue



        upos, upos_ok = _majority_value([tok_a.upos, tok_b.upos, tok_c.upos])

        lemma, lemma_ok = _majority_value([tok_a.lemma, tok_b.lemma, tok_c.lemma])

        head, head_ok = _majority_value([tok_a.head, tok_b.head, tok_c.head])

        deprel, deprel_ok = _majority_value([tok_a.deprel, tok_b.deprel, tok_c.deprel])

        feats_tuple, feats_ok = _majority_value(

            [tuple(sorted(tok_a.feats.items())), tuple(sorted(tok_b.feats.items())), tuple(sorted(tok_c.feats.items()))]

        )



        if not all([upos_ok, lemma_ok, head_ok, deprel_ok, feats_ok]):

            disagreements.append(

                {

                    "token_index": idx,

                    "token_id": tok_a.id,

                    "form": tok_a.form,

                    "values": {

                        "upos": [tok_a.upos, tok_b.upos, tok_c.upos],

                        "lemma": [tok_a.lemma, tok_b.lemma, tok_c.lemma],

                        "head": [tok_a.head, tok_b.head, tok_c.head],

                        "deprel": [tok_a.deprel, tok_b.deprel, tok_c.deprel],

                    },

                }

            )

            continue



        voted_tokens.append(

            UDTokenAnnotation(

                id=tok_a.id,

                form=tok_a.form,

                lemma=lemma,

                upos=upos,

                xpos=tok_a.xpos if tok_a.xpos == tok_b.xpos else "_",

                feats=dict(feats_tuple),

                head=head,

                deprel=deprel,

                misc={},

                katharevousa=tok_a.katharevousa,

            )

        )



    if disagreements:

        return VoteResult(

            auto_accepted=None,

            disagreement_payload={

                "sent_id": base.sentence_id,

                "text": base.text,

                "reason": "field_disagreement",

                "disagreements": disagreements,

                "candidates": [

                    {"model": a.metadata.get("annotator"), "conllu": a.to_conllu()}

                    for a in by_model

                ],

            },

        )



    voted_sentence = UDSentenceAnnotation(

        sentence_id=base.sentence_id,

        text=base.text,

        tokens=voted_tokens,

        metadata={"vote": "majority_2of3"},

    )

    return VoteResult(auto_accepted=voted_sentence, disagreement_payload=None)

