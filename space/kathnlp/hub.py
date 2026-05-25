"""Hugging Face Hub helpers for kathnlp dependency parsers.

Public API:

    from kathnlp.hub import load_from_hub

    parser = load_from_hub("gmikros/kathnlp-xlmr")
    parsed = parser.parse("Ἡ Κυβέρνησις παρακαλεῖται νά ἀποδεχθῇ τό αἴτημα.")
    for tok in parsed:
        print(tok.id, tok.form, tok.upos, tok.head, tok.deprel)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from kathnlp.schema import UDSentenceAnnotation, UDTokenAnnotation
from kathnlp.training.transformer_parser import (
    ParserLabelMaps,
    TransformerDependencyParser,
    predict_sentences,
)


@dataclass
class ParsedToken:
    """One token of a parsed sentence."""

    id: int
    form: str
    upos: str
    head: int
    deprel: str


class Parser:
    """A loaded kathnlp dependency parser ready for inference."""

    def __init__(
        self,
        model: TransformerDependencyParser,
        tokenizer,
        label_maps: ParserLabelMaps,
        max_length: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.label_maps = label_maps
        self.max_length = max_length

    def parse(self, text: str) -> List[ParsedToken]:
        """Whitespace-tokenize *text* and return one ParsedToken per word.

        For more controlled tokenization, pre-split your text and join with
        single spaces.
        """
        forms = text.split()
        if not forms:
            return []
        ud_tokens = [
            UDTokenAnnotation(
                id=i + 1,
                form=form,
                lemma=form,
                upos="X",
                xpos=None,
                feats={},
                head=0,
                deprel="dep",
                deps=None,
                misc={},
            )
            for i, form in enumerate(forms)
        ]
        sentence = UDSentenceAnnotation(
            sent_id="user_input",
            text=text,
            tokens=ud_tokens,
        )
        pred_upos, pred_deprel, pred_heads = predict_sentences(
            self.model,
            self.tokenizer,
            [sentence],
            self.label_maps,
            max_length=self.max_length,
        )
        return [
            ParsedToken(
                id=i + 1,
                form=forms[i],
                upos=pred_upos[i],
                head=int(pred_heads[i]),
                deprel=pred_deprel[i],
            )
            for i in range(len(forms))
        ]


def load_from_hub(
    repo_id: str,
    revision: Optional[str] = None,
    device: str = "cpu",
    cache_dir: Optional[str] = None,
) -> Parser:
    """Download and load a kathnlp dependency parser from the Hugging Face Hub.

    Parameters
    ----------
    repo_id : str
        e.g. ``"gmikros/kathnlp-xlmr"``.
    revision : str, optional
        Branch, tag, or commit hash. Defaults to the latest ``main``.
    device : str, default ``"cpu"``
        Torch device for inference. Use ``"cuda"`` if a GPU is available.
    cache_dir : str, optional
        Override the default Hugging Face Hub cache directory.

    Returns
    -------
    Parser
        Ready-to-use parser whose ``.parse(text)`` returns a list of
        ``ParsedToken``.
    """
    snapshot = Path(
        snapshot_download(repo_id=repo_id, revision=revision, cache_dir=cache_dir)
    )
    metadata = json.loads((snapshot / "metadata.json").read_text(encoding="utf-8"))

    encoder_name = metadata["train_config"]["encoder_name"]
    upos_labels = metadata["upos_labels"]
    deprel_labels = metadata["deprel_labels"]
    max_length = int(metadata["train_config"].get("max_length", 256))

    # Build the architecture; the encoder is initialised from the base
    # checkpoint on HF (xlm-roberta-base), then overwritten with the
    # fine-tuned weights below.
    model = TransformerDependencyParser(
        encoder_name=encoder_name,
        upos_count=len(upos_labels),
        deprel_count=len(deprel_labels),
    )

    # The fine-tuned encoder weights live in encoder/model.safetensors.
    encoder_dir = snapshot / "encoder"
    if encoder_dir.exists():
        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(str(encoder_dir))
        model.encoder.load_state_dict(encoder.state_dict())

    # Parser-specific heads live in parser_heads.pt. We load with
    # strict=False so a heads-only file (just the non-encoder params) and
    # a full state-dict file (encoder + heads) both work.
    parser_heads_path = snapshot / "parser_heads.pt"
    if parser_heads_path.exists():
        state = torch.load(parser_heads_path, map_location="cpu")
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(str(snapshot / "tokenizer"))
    label_maps = ParserLabelMaps(
        upos_to_id={label: i for i, label in enumerate(upos_labels)},
        upos_labels=upos_labels,
        deprel_to_id={label: i for i, label in enumerate(deprel_labels)},
        deprel_labels=deprel_labels,
    )
    return Parser(model, tokenizer, label_maps, max_length=max_length)


__all__ = ["Parser", "ParsedToken", "load_from_hub"]
