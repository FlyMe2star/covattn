"""Small helpers for the booktabs tables: number formatting and marking the best entry per column.

Convention used in every table: the best value per column is bold, judged only over the rows that are passed
as candidates (reference rows such as ARX are excluded by the caller). Ties after rounding are all bold.
Modes: "min" (lower is better), "max" (higher is better), "one" (closest to 1 is better, used for g).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

UP, DOWN, TO1 = "$\\uparrow$", "$\\downarrow$", "$\\to$1"


def esc(s: str) -> str:
    return s.replace("_", "\\_").replace("%", "\\%")


def _score(v: float, mode: str) -> float:
    return {"min": v, "max": -v, "one": abs(v - 1.0)}[mode]


def mark_best(vals: Sequence[float], fmt: str, mode: str | None, candidates: Sequence[bool] | None = None,
              suffix: Sequence[str] | None = None) -> list[str]:
    """Format `vals` with `fmt`; bold the best among `candidates` (compared after rounding to the printed digits)."""
    txt = [("--" if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, fmt)) for v in vals]
    cand = list(candidates) if candidates is not None else [True] * len(vals)
    if mode is not None:
        sc = [(_score(float(t), mode) if (c and t != "--") else math.inf) for t, c in zip(txt, cand)]
        best = min(sc)
        if math.isfinite(best):
            txt = [f"\\textbf{{{t}}}" if s == best else t for t, s in zip(txt, sc)]
    if suffix is not None:
        txt = [t + s for t, s in zip(txt, suffix)]
    return txt


def rows_to_tex(colspec: str, header: list[str], rows: list[list[str] | str]) -> str:
    """rows: list of cell lists, or the literal strings 'midrule' / any raw LaTeX line starting with a backslash."""
    out = [f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", *header, "\\midrule"]
    for r in rows:
        out.append(r if isinstance(r, str) else " & ".join(r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(out) + "\n"
