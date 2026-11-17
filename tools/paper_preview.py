"""Typeset every table and figure in the LNCS page (llncs class) to check width and legibility.

Writes reports/preview/preview.tex, compiles it with pdflatex (needs llncs.cls, booktabs) and, if pdftoppm is
available, renders one PNG per page. Captions are placeholders (file names only); this is a layout check, not text.

  python tools/paper_preview.py
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from covattn.io_utils import results_root

TABLES = ["table1_synthetic", "table2_real_cov", "table3_univariate", "table4_ablations", "table4b_sensitivity",
          "table5_fev", "table6_semisynth", "table7_seeds", "table8_other_models"]
FIGS = ["fig1_gain_by_family", "fig2_examples", "fig3_restoration", "fig4_reg_sink", "fig5_ablations",
        "fig6_other_models", "fig7_real_world"]


def main() -> None:
    rep = results_root() / "reports"
    out = rep / "preview"
    out.mkdir(parents=True, exist_ok=True)
    body = []
    for t in TABLES:
        f = rep / "tables" / f"{t}.tex"
        if f.exists():
            body += ["\\begin{table}[p]", "\\centering", f"\\caption{{\\texttt{{{t.replace('_', chr(92) + '_')}}}}}",
                     f"\\input{{{f.as_posix()}}}", "\\end{table}"]
    for g in FIGS:
        f = rep / "figs" / f"{g}.pdf"
        if f.exists():
            body += ["\\begin{figure}[p]", "\\centering", f"\\includegraphics{{{f.as_posix()}}}",
                     f"\\caption{{\\texttt{{{g.replace('_', chr(92) + '_')}}}}}", "\\end{figure}"]
    doc = ["\\documentclass[runningheads]{llncs}", "\\usepackage[T1]{fontenc}", "\\usepackage{graphicx}",
           "\\usepackage{booktabs}", "\\usepackage{multirow}", "\\usepackage{amssymb}", "\\begin{document}",
           "\\title{Layout preview}", "\\author{Anonymous}", "\\institute{Anonymous}", "\\maketitle",
           "Text width: \\the\\textwidth.", *body, "\\end{document}"]
    tex = out / "preview.tex"
    tex.write_text("\n".join(doc) + "\n", encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex.name], cwd=out,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    logf = (out / "preview.log").read_text(encoding="utf-8", errors="replace") if (out / "preview.log").exists() else ""
    over = [ln for ln in logf.splitlines() if ln.startswith("Overfull \\hbox")]
    print("pdflatex", "OK" if r.returncode == 0 else f"FAILED ({r.returncode})")
    print("\n".join(over) if over else "no overfull boxes")
    if r.returncode != 0:
        print(r.stdout[-3000:])
        return
    if shutil.which("pdftoppm"):
        for old in out.glob("page-*.png"):
            old.unlink()
        subprocess.run(["pdftoppm", "-png", "-r", "110", "preview.pdf", "page"], cwd=out, check=False)
        print("pages:", ", ".join(sorted(p.name for p in out.glob("page-*.png"))))


if __name__ == "__main__":
    main()
