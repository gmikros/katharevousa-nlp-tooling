from __future__ import annotations

import re
from dataclasses import dataclass


GREEK_LETTER = r"[\u0370-\u03ff\u1f00-\u1fff]"
LOWER_GREEK_LETTER = r"[α-ωάέήίόύώἀ-῾]"

HARD_HYPHEN_BREAK_RE = re.compile(
    rf"({GREEK_LETTER}{{2,}})\s*[-¬]\s*\n\s*({LOWER_GREEK_LETTER}{GREEK_LETTER}{{1,}})",
    flags=re.IGNORECASE,
)
SOFT_WRAP_RE = re.compile(r"[ \t]*\n[ \t]*")
MULTISPACE_RE = re.compile(r"\s{2,}")


@dataclass
class ReconstructionStats:
    rows_total: int = 0
    rows_changed: int = 0
    join_repairs: int = 0
    newline_repairs: int = 0


def reconstruct_text(text: str) -> tuple[str, dict]:
    source = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    if not source.strip():
        return "", {"join_repairs": 0, "newline_repairs": 0, "changed": False}

    join_repairs = 0

    def _join_break(match: re.Match) -> str:
        nonlocal join_repairs
        join_repairs += 1
        return f"{match.group(1)}{match.group(2)}"

    repaired = HARD_HYPHEN_BREAK_RE.sub(_join_break, source)
    repaired = repaired.replace("\u00ad", "")
    newline_repairs = repaired.count("\n")
    repaired = SOFT_WRAP_RE.sub(" ", repaired)
    repaired = MULTISPACE_RE.sub(" ", repaired).strip()

    return repaired, {
        "join_repairs": join_repairs,
        "newline_repairs": newline_repairs,
        "changed": repaired != source.strip(),
    }

