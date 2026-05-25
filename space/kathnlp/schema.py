from __future__ import annotations



from dataclasses import dataclass, field

from typing import Any



UD_UPOS = {

    "ADJ",

    "ADP",

    "ADV",

    "AUX",

    "CCONJ",

    "DET",

    "INTJ",

    "NOUN",

    "NUM",

    "PART",

    "PRON",

    "PROPN",

    "PUNCT",

    "SCONJ",

    "SYM",

    "VERB",

    "X",

}



UD_DEPREL = {

    "acl",

    "advcl",

    "advmod",

    "amod",

    "appos",

    "aux",

    "case",

    "cc",

    "ccomp",

    "clf",

    "compound",

    "conj",

    "cop",

    "csubj",

    "dep",

    "det",

    "discourse",

    "dislocated",

    "expl",

    "fixed",

    "flat",

    "goeswith",

    "iobj",

    "list",

    "mark",

    "nmod",

    "nsubj",

    "nummod",

    "obj",

    "obl",

    "orphan",

    "parataxis",

    "preconj",

    "punct",

    "reparandum",

    "root",

    "vocative",

    "xcomp",

}



UD_FEAT_KEYS = {

    "Abbr",

    "Animacy",

    "Aspect",

    "Case",

    "Definite",

    "Degree",

    "Evident",

    "Foreign",

    "Gender",

    "Mood",

    "Number",

    "NumForm",

    "NumType",

    "Person",

    "Polarity",

    "Polite",

    "Poss",

    "PronType",

    "Reflex",

    "Tense",

    "Typo",

    "VerbForm",

    "Voice",

}





@dataclass

class KatharevousaExtensions:

    archaic_lexeme_class: str | None = None

    orthography_source: str | None = None

    legacy_morphology_flags: list[str] = field(default_factory=list)

    legal_register_markers: list[str] = field(default_factory=list)

    notes: str | None = None





@dataclass

class UDTokenAnnotation:

    id: int

    form: str

    lemma: str

    upos: str

    xpos: str = "_"

    feats: dict[str, str] = field(default_factory=dict)

    head: int = 0

    deprel: str = "dep"

    deps: str = "_"

    misc: dict[str, str] = field(default_factory=dict)

    katharevousa: KatharevousaExtensions | None = None



    def validate(self) -> None:

        if self.id < 1:

            raise ValueError("Token id must be >= 1")

        if self.upos not in UD_UPOS:

            raise ValueError(f"Invalid UPOS: {self.upos}")

        deprel_base = self.deprel.split(":", 1)[0]

        if deprel_base not in UD_DEPREL:

            raise ValueError(f"Invalid DEPREL: {self.deprel}")

        invalid_feat_keys = [k for k in self.feats if k not in UD_FEAT_KEYS]

        if invalid_feat_keys:

            raise ValueError(f"Invalid FEATS keys: {invalid_feat_keys}")



    def feats_as_conllu(self) -> str:

        if not self.feats:

            return "_"

        return "|".join(f"{k}={v}" for k, v in sorted(self.feats.items()))



    def misc_as_conllu(self) -> str:

        if not self.misc:

            return "_"

        return "|".join(f"{k}={v}" for k, v in sorted(self.misc.items()))



    def to_conllu_row(self) -> str:

        self.validate()

        return "\t".join(

            [

                str(self.id),

                self.form,

                self.lemma,

                self.upos,

                self.xpos,

                self.feats_as_conllu(),

                str(self.head),

                self.deprel,

                self.deps,

                self.misc_as_conllu(),

            ]

        )



    def to_sidecar(self) -> dict[str, Any]:

        payload: dict[str, Any] = {"token_id": self.id}

        if self.katharevousa is None:

            return payload

        payload["katharevousa"] = {

            "archaic_lexeme_class": self.katharevousa.archaic_lexeme_class,

            "orthography_source": self.katharevousa.orthography_source,

            "legacy_morphology_flags": self.katharevousa.legacy_morphology_flags,

            "legal_register_markers": self.katharevousa.legal_register_markers,

            "notes": self.katharevousa.notes,

        }

        return payload





@dataclass

class UDSentenceAnnotation:

    sentence_id: str

    text: str

    tokens: list[UDTokenAnnotation]

    metadata: dict[str, str] = field(default_factory=dict)



    def validate(self) -> None:

        if not self.tokens:

            raise ValueError("Sentence must include at least one token")

        token_ids = {token.id for token in self.tokens}

        if token_ids != set(range(1, len(self.tokens) + 1)):

            raise ValueError("Token IDs must be contiguous and start at 1")

        for token in self.tokens:

            token.validate()

            if token.head not in token_ids and token.head != 0:

                raise ValueError(f"Token {token.id} has invalid head {token.head}")

        if sum(1 for token in self.tokens if token.head == 0) != 1:

            raise ValueError("Sentence must have exactly one syntactic root")



    def to_conllu(self) -> str:

        self.validate()

        header = [f"# sent_id = {self.sentence_id}", f"# text = {self.text}"]

        header.extend(f"# {k} = {v}" for k, v in sorted(self.metadata.items()))

        body = [token.to_conllu_row() for token in self.tokens]

        return "\n".join(header + body)



    def to_sidecar(self) -> dict[str, Any]:

        return {

            "sent_id": self.sentence_id,

            "text": self.text,

            "metadata": self.metadata,

            "tokens": [token.to_sidecar() for token in self.tokens],

        }

