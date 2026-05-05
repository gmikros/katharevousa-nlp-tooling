from __future__ import annotations



from dataclasses import dataclass

from pathlib import Path



from kathnlp.pipelines.serialization import parse_conllu_sentence

from kathnlp.schema import UDSentenceAnnotation





@dataclass

class TokenExample:

    sent_id: str

    token_id: int

    form: str

    prev_form: str

    next_form: str

    sentence_len: int

    upos: str

    deprel: str

    head: int





def load_conllu(path: Path) -> list[UDSentenceAnnotation]:

    text = path.read_text(encoding="utf-8").strip()

    if not text:

        return []

    blocks = text.split("\n\n")

    return [parse_conllu_sentence(block) for block in blocks if block.strip()]





def sentence_to_examples(sentence: UDSentenceAnnotation) -> list[TokenExample]:

    examples: list[TokenExample] = []

    tokens = sentence.tokens

    sentence_len = len(tokens)

    for i, tok in enumerate(tokens):

        prev_tok = tokens[i - 1].form if i > 0 else "<BOS>"

        next_tok = tokens[i + 1].form if i + 1 < len(tokens) else "<EOS>"

        examples.append(

            TokenExample(

                sent_id=sentence.sentence_id,

                token_id=tok.id,

                form=tok.form,

                prev_form=prev_tok,

                next_form=next_tok,

                sentence_len=sentence_len,

                upos=tok.upos,

                deprel=tok.deprel,

                head=tok.head,

            )

        )

    return examples





def sentences_to_examples(sentences: list[UDSentenceAnnotation]) -> list[TokenExample]:

    flat: list[TokenExample] = []

    for sentence in sentences:

        flat.extend(sentence_to_examples(sentence))

    return flat

