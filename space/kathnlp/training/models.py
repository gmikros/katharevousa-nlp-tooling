from __future__ import annotations



from dataclasses import dataclass

from typing import Any



import joblib

from sklearn.feature_extraction import DictVectorizer

from sklearn.linear_model import LogisticRegression



from kathnlp.training.dataset import TokenExample





def _features(example: TokenExample) -> dict[str, Any]:

    position_bucket = min(10, int((example.token_id / max(1, example.sentence_len)) * 10))

    return {

        "form": example.form.lower(),

        "prefix2": example.form[:2].lower(),

        "prefix3": example.form[:3].lower(),

        "suffix3": example.form[-3:].lower(),

        "suffix4": example.form[-4:].lower(),

        "prev": example.prev_form.lower(),

        "next": example.next_form.lower(),

        "sentence_len": min(example.sentence_len, 80),

        "position_bucket": position_bucket,

        "has_apostrophe": "'" in example.form or "’" in example.form,

        "has_hyphen": "-" in example.form,

        "is_punct_like": not any(ch.isalnum() for ch in example.form),

        "is_upper": example.form.isupper(),

        "is_title": example.form[:1].isupper(),

        "is_digit": example.form.isdigit(),

    }





def _head_label(example: TokenExample) -> str:

    if example.head == 0:

        return "ROOT"

    offset = example.head - example.token_id

    if offset > 20:

        return "R>20"

    if offset < -20:

        return "L>20"

    return str(offset)





def _decode_head_label(label: str, example: TokenExample) -> int:

    if label == "ROOT":

        return 0

    if label == "R>20":

        return min(example.sentence_len, example.token_id + 21)

    if label == "L>20":

        return max(1, example.token_id - 21)

    try:

        head = example.token_id + int(label)

    except ValueError:

        return 0

    if head <= 0:

        return 0

    if head > example.sentence_len:

        return example.sentence_len

    # Self-loops are invalid in basic dependencies.

    if head == example.token_id:

        return 0

    return head





@dataclass

class TokenTagger:

    vectorizer: DictVectorizer

    upos_model: LogisticRegression

    deprel_model: LogisticRegression

    head_model: LogisticRegression



    def predict(self, examples: list[TokenExample]) -> tuple[list[str], list[str], list[int]]:

        x = self.vectorizer.transform([_features(ex) for ex in examples])

        pred_upos = self.upos_model.predict(x).tolist()

        pred_deprel = self.deprel_model.predict(x).tolist()

        pred_head_labels = self.head_model.predict(x).tolist()

        pred_heads = [

            _decode_head_label(label, ex) for label, ex in zip(pred_head_labels, examples)

        ]

        return pred_upos, pred_deprel, pred_heads



    def save(self, path: str) -> None:

        joblib.dump(self, path)



    @staticmethod

    def load(path: str) -> "TokenTagger":

        return joblib.load(path)





def train_token_tagger(examples: list[TokenExample]) -> TokenTagger:

    vectorizer = DictVectorizer(sparse=True)

    x = vectorizer.fit_transform([_features(ex) for ex in examples])

    y_upos = [ex.upos for ex in examples]

    y_deprel = [ex.deprel for ex in examples]

    y_head = [_head_label(ex) for ex in examples]



    upos_model = LogisticRegression(max_iter=2000, n_jobs=None)

    deprel_model = LogisticRegression(max_iter=2000, n_jobs=None)

    head_model = LogisticRegression(max_iter=2500, n_jobs=None)

    upos_model.fit(x, y_upos)

    deprel_model.fit(x, y_deprel)

    head_model.fit(x, y_head)



    return TokenTagger(

        vectorizer=vectorizer,

        upos_model=upos_model,

        deprel_model=deprel_model,

        head_model=head_model,

    )

