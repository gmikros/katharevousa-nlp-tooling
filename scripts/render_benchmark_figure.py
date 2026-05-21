"""Render the benchmark comparison figure used in the paper and the README.

Outputs paper/figures/benchmark_metrics.png at publication-quality DPI.
Run from the repository root:

    python scripts/render_benchmark_figure.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODELS = [
    "spaCy\nGreek",
    "Stanza\nAncient Greek\n(PROIEL)",
    "Stanza\nGreek\n(pretrained)",
    "mBERT",
    "Stanza\n(custom)",
    "Feature-based\nbaseline",
    "XLM-R\n(release)",
]

METRICS = ["UPOS", "DEPREL F1", "UAS", "LAS"]

# Scores reported in reports/*.json and the manuscript.
SCORES = np.array(
    [
        [0.6721, 0.5315, 0.5492, 0.4183],  # spaCy Greek
        [0.5292, 0.4044, 0.4850, 0.3076],  # Stanza Ancient Greek PROIEL
        [0.6125, 0.4242, 0.6079, 0.3396],  # Stanza Greek pretrained
        [0.8260, 0.6076, 0.5886, 0.4537],  # mBERT
        [0.7694, 0.6588, 0.5756, 0.4943],  # Stanza custom
        [0.9040, 0.7451, 0.5781, 0.5072],  # Feature-based baseline
        [0.8893, 0.7250, 0.6098, 0.5162],  # XLM-R release candidate
    ]
)

COLORS = ["#4C78A8", "#59A14F", "#F28E2B", "#B07AA1"]


def render(output_path: Path) -> None:
    n_models, n_metrics = SCORES.shape
    bar_width = 0.18
    group_centers = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.07, right=0.985)

    for metric_idx, metric_name in enumerate(METRICS):
        offsets = (metric_idx - (n_metrics - 1) / 2) * bar_width
        bars = ax.bar(
            group_centers + offsets,
            SCORES[:, metric_idx],
            width=bar_width,
            label=metric_name,
            color=COLORS[metric_idx],
            edgecolor="white",
            linewidth=0.6,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.012,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(MODELS, fontsize=9)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_ylabel("Score (higher is better)", fontsize=10)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", color="#DDDDDD", linewidth=0.7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", which="both", length=0, pad=6)

    ax.set_title(
        "Fixed-split benchmark comparison",
        fontsize=13,
        loc="left",
        pad=18,
        fontweight="semibold",
    )
    ax.text(
        0.0,
        1.015,
        "Held-out test set: 340 sentences / 4,093 tokens. Seed 42.",
        transform=ax.transAxes,
        fontsize=9,
        color="#555555",
    )

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=n_metrics,
        frameon=False,
        fontsize=10,
        handlelength=1.4,
        columnspacing=2.2,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    output_path = repo_root / "paper" / "figures" / "benchmark_metrics.png"
    render(output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
