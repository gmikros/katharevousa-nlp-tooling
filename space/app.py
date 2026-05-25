"""Gradio demo for the kathnlp Katharevousa Greek dependency parser.

Loads `gmikros/kathnlp-xlmr` from the Hugging Face Hub and lets visitors
paste a Katharevousa sentence, returning UPOS tags, dependency arcs, and
a CoNLL-U-style table.
"""

from __future__ import annotations

import functools
import html
import os
from typing import List

import gradio as gr
import torch

from kathnlp.hub import ParsedToken, load_from_hub


MODEL_REPO = os.environ.get("KATHNLP_MODEL_REPO", "gmikros/kathnlp-xlmr")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


EXAMPLES = [
    [
        "Ἡ Κυβέρνησις παρακαλεῖται νά ἀποδεχθῇ τό αἴτημα τοῦ χωρίου Πολυλόφου "
        "περί διατηρήσεως τοῦ Δημοτικοῦ Σχολείου."
    ],
    [
        "Πρέπει ἡ Κυβέρνησις νά ἐνδιαφερθῇ διά τήν Κοινότητα Ἁγίου Γεωργίου "
        "Αὐλωναρίου καί νά χορηγήσῃ μίαν σοβαράν οἰκονομικήν ἐνίσχυσιν."
    ],
    [
        "Τό ὑπουργεῖον Ἐσωτερικῶν ἐρωτᾶται διά τήν λῆψιν τῶν ἀναγκαίων μέτρων "
        "ὑπέρ τῶν πληγέντων κατοίκων τῆς περιοχῆς."
    ],
]


@functools.lru_cache(maxsize=1)
def get_parser():
    """Load the model once and cache the Parser object."""
    return load_from_hub(MODEL_REPO, device=DEVICE)


def _render_conllu_table(tokens: List[ParsedToken]) -> str:
    rows = "".join(
        f"<tr>"
        f"<td>{tok.id}</td>"
        f"<td><b>{html.escape(tok.form)}</b></td>"
        f"<td>{tok.upos}</td>"
        f"<td>{tok.head}</td>"
        f"<td>{tok.deprel}</td>"
        f"</tr>"
        for tok in tokens
    )
    return (
        "<table style='width:100%;border-collapse:collapse;font-family:"
        "ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px'>"
        "<thead><tr style='background:#f3f4f6;text-align:left'>"
        "<th style='padding:6px 8px'>ID</th>"
        "<th style='padding:6px 8px'>FORM</th>"
        "<th style='padding:6px 8px'>UPOS</th>"
        "<th style='padding:6px 8px'>HEAD</th>"
        "<th style='padding:6px 8px'>DEPREL</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def _render_arc_diagram(tokens: List[ParsedToken]) -> str:
    """Render a simple SVG dependency diagram."""
    if not tokens:
        return ""

    char_width = 11
    row_height = 36
    margin = 24
    word_top = margin + 80
    deprel_top = word_top + 22
    arc_base = word_top - 6

    word_centers: List[float] = []
    x = margin
    for tok in tokens:
        width = max(len(tok.form), len(tok.upos)) * char_width + 16
        cx = x + width / 2
        word_centers.append(cx)
        x += width + 12
    total_width = max(int(x + margin), 480)
    total_height = deprel_top + 24

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{total_width}' "
        f"height='{total_height + 40}' style='background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;font-family:ui-sans-serif,system-ui,sans-serif'>"
    ]

    # Arcs.
    for tok in tokens:
        if tok.head == 0:
            head_cx = word_centers[tok.id - 1]
            parts.append(
                f"<text x='{head_cx}' y='{margin}' text-anchor='middle' "
                f"font-size='11' fill='#6b7280'>ROOT</text>"
            )
            parts.append(
                f"<line x1='{head_cx}' y1='{margin + 4}' x2='{head_cx}' "
                f"y2='{arc_base}' stroke='#0f766e' stroke-width='1.5'/>"
            )
            continue
        src_cx = word_centers[tok.head - 1]
        dst_cx = word_centers[tok.id - 1]
        distance = abs(dst_cx - src_cx)
        peak_y = max(margin + 4, arc_base - max(28, distance * 0.35))
        mid_x = (src_cx + dst_cx) / 2
        parts.append(
            f"<path d='M{src_cx},{arc_base} Q{mid_x},{peak_y} {dst_cx},{arc_base}' "
            f"fill='none' stroke='#1f2937' stroke-width='1.2' opacity='0.6'/>"
        )
        parts.append(
            f"<text x='{mid_x}' y='{peak_y - 4}' text-anchor='middle' "
            f"font-size='10' fill='#374151'>{html.escape(tok.deprel)}</text>"
        )
        parts.append(
            f"<polygon points='{dst_cx - 4},{arc_base - 6} {dst_cx + 4},"
            f"{arc_base - 6} {dst_cx},{arc_base}' fill='#1f2937' opacity='0.6'/>"
        )

    # Words + UPOS.
    for tok, cx in zip(tokens, word_centers):
        parts.append(
            f"<text x='{cx}' y='{word_top}' text-anchor='middle' "
            f"font-size='15' font-weight='600' fill='#111827'>"
            f"{html.escape(tok.form)}</text>"
        )
        parts.append(
            f"<text x='{cx}' y='{deprel_top}' text-anchor='middle' "
            f"font-size='11' fill='#6b7280'>{html.escape(tok.upos)}</text>"
        )

    parts.append("</svg>")
    return "".join(parts)


def parse(text: str):
    text = (text or "").strip()
    if not text:
        return "<p style='color:#6b7280'>Type a Katharevousa sentence above and press <b>Parse</b>.</p>", ""
    parser = get_parser()
    tokens = parser.parse(text)
    if not tokens:
        return "<p style='color:#6b7280'>No tokens parsed.</p>", ""
    return _render_arc_diagram(tokens), _render_conllu_table(tokens)


DESCRIPTION = """
# kathnlp · Katharevousa Greek dependency parser

A Universal-Dependencies-style parser for **Katharevousa Greek**, the archaizing
official register of 20th-century Greek law, administration, and parliament.

- 📄 Paper: [arXiv:2605.22978](https://arxiv.org/abs/2605.22978)
- 🧠 Model: [`gmikros/kathnlp-xlmr`](https://huggingface.co/gmikros/kathnlp-xlmr)
- 🗂️ Treebank: [`gmikros/kathnlp-treebank`](https://huggingface.co/datasets/gmikros/kathnlp-treebank)
- 💻 Code: [github.com/gmikros/katharevousa-nlp-tooling](https://github.com/gmikros/katharevousa-nlp-tooling)

> The first request after a cold start downloads the ~1.1 GB model and may take
> 30–60 seconds. Subsequent parses are fast.
"""


with gr.Blocks(
    title="kathnlp · Katharevousa Greek dependency parser",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="Katharevousa sentence",
                placeholder="Παρακαλεῖται ἡ Κυβέρνησις νά …",
                lines=3,
            )
            parse_btn = gr.Button("Parse", variant="primary")
            gr.Examples(EXAMPLES, inputs=[input_text], label="Examples")
        with gr.Column(scale=4):
            tree_view = gr.HTML(label="Dependency tree")
            table_view = gr.HTML(label="CoNLL-U output")

    parse_btn.click(parse, inputs=[input_text], outputs=[tree_view, table_view])
    input_text.submit(parse, inputs=[input_text], outputs=[tree_view, table_view])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
