from __future__ import annotations



import json

from pathlib import Path



from kathnlp.schema import UDSentenceAnnotation

from kathnlp.schema import UDTokenAnnotation





def write_conllu(sentences: list[UDSentenceAnnotation], output_path: Path) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = "\n\n".join(sentence.to_conllu() for sentence in sentences) + "\n"

    output_path.write_text(payload, encoding="utf-8")





def write_sidecar_json(

    sentences: list[UDSentenceAnnotation], output_path: Path

) -> None:

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = [sentence.to_sidecar() for sentence in sentences]

    output_path.write_text(

        json.dumps(payload, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )





def parse_conllu_sentence(block: str) -> UDSentenceAnnotation:

    lines = [line.strip() for line in block.splitlines() if line.strip()]

    sent_id = ""

    text = ""

    metadata: dict[str, str] = {}

    tokens: list[UDTokenAnnotation] = []

    for line in lines:

        if line.startswith("#"):

            key, _, value = line[1:].partition("=")

            key = key.strip()

            value = value.strip()

            if key == "sent_id":

                sent_id = value

            elif key == "text":

                text = value

            else:

                metadata[key] = value

            continue

        parts = line.split("\t")

        if len(parts) != 10:

            continue

        feats = {}

        misc = {}

        if parts[5] != "_":

            for piece in parts[5].split("|"):

                k, _, v = piece.partition("=")

                feats[k] = v

        if parts[9] != "_":

            for piece in parts[9].split("|"):

                k, _, v = piece.partition("=")

                misc[k] = v

        tokens.append(

            UDTokenAnnotation(

                id=int(parts[0]),

                form=parts[1],

                lemma=parts[2],

                upos=parts[3],

                xpos=parts[4],

                feats=feats,

                head=int(parts[6]),

                deprel=parts[7],

                deps=parts[8],

                misc=misc,

            )

        )

    return UDSentenceAnnotation(

        sentence_id=sent_id,

        text=text,

        tokens=tokens,

        metadata=metadata,

    )

