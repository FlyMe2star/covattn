"""Regenerate colab/run_all.ipynb (keeps the notebook diff-able as code).

Open on Colab: https://colab.research.google.com/github/FlyMe2star/covattn/blob/main/colab/run_all.ipynb
"""
import json
from pathlib import Path

REPO = "https://github.com/FlyMe2star/covattn.git"

cells = [
    ("md", "# covattn - run experiments on Colab\n"
           "Runtime -> Change runtime type -> **A100 GPU**. Run cells 1-5 once per session; every experiment cell "
           "uses `--resume`, so after a disconnect just re-run 1-5 and the experiment cell.\n\n"
           "Scripts run as subprocesses (`!python ...`), so pinned packages take effect without restarting the runtime."),
    ("code", "# 1. mount Drive (results live here)\nfrom google.colab import drive\ndrive.mount('/content/drive')"),
    ("code", "# 2. get / update the code from GitHub\n"
             f"REPO = '{REPO}'\nBRANCH = 'main'\n"
             "import os\n"
             "if not os.path.isdir('/content/covattn/.git'):\n"
             "    !git clone -q -b {BRANCH} {REPO} /content/covattn\n"
             "else:\n"
             "    !git -C /content/covattn fetch -q origin && git -C /content/covattn reset -q --hard origin/{BRANCH}\n"
             "%cd /content/covattn\n"
             "!git log --oneline -3"),
    ("code", "# 3. install pinned dependencies (keeps Colab's own CUDA torch)\n"
             "!pip install -q -r requirements.txt && pip install -q -e . --no-deps"),
    ("code", "# 4. GPU check\n!nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv\n"
             "!python -c \"import torch, chronos; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), "
             "torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-', '| chronos', chronos.__version__)\""),
    ("code", "# 5. results directory on Drive\n"
             "os.environ['RESULTS_DIR'] = '/content/drive/MyDrive/pakdd27/results'\n"
             "os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'\n"
             "os.makedirs(os.environ['RESULTS_DIR'], exist_ok=True)\n"
             "print(os.environ['RESULTS_DIR'])"),
    ("md", "## Smoke test (about 2 min): unit + model tests, source checks, Exp1 quick"),
    ("code", "!python -m pytest -q\n!python scripts/verify_model.py > /dev/null && echo verify_model OK\n"
             "!python scripts/exp1_phenomenon.py --quick --resume --run-name colab_quick 2>&1 | grep -E 'ESTIMATED|done|Error|Traceback'"),
    ("md", "## Stage 0 - Exp1 (phenomenon)\n"
           "The canonical run `exp1/main` (304 cells) is already on Drive; this cell only checks it."),
    ("code", "!ls $RESULTS_DIR/exp1/main && echo \"cells: $(ls $RESULTS_DIR/exp1/main/cells | wc -l)\"\n"
             "!python scripts/exp1_phenomenon.py --summary-only && head -30 $RESULTS_DIR/reports/exp1_summary_main.md"),
    ("code", "# complete / re-run Exp1 on this GPU (resumable; skips finished cells)\n"
             "# !python scripts/exp1_phenomenon.py --resume --batch-size 256"),
    ("md", "## Stage 1 - Exp2 (mechanism: decomposition, patching, REG sink, cross-gain preview)\n"
           "Val split. 48 cells = 6 systems x L{512,2048} x {prbs, multisine} x {step, pulse}, 64 pairs each. "
           "The script prints its own time estimate for this GPU (expect roughly 1 h on L4)."),
    ("code", "!python -m pytest -q tests/test_hooks.py\n"
             "!python scripts/exp2_mechanism.py --quick --resume --run-name colab_quick 2>&1 | grep -E 'ESTIMATED|done|Error|Traceback'"),
    ("code", "!python scripts/exp2_mechanism.py --resume 2>&1 | grep -v -E 'Loading weights|UserWarning|warnings.warn'"),
    ("code", "!head -120 $RESULTS_DIR/reports/exp2_summary_main.md"),
    ("md", "## Stage 2 - Exp3 (repairs vs baselines, synthetic test) and Exp4 (real covariate + univariate)\n"
           "Exp3 runs: R1 selection on val -> dither selection -> R2 gate learning -> full FT + LoRA (official "
           "`fit`) -> test evaluation (152 cells). Each stage is resumable. Exp4 reuses the Exp3 artefacts."),
    ("code", "!python -m pytest -q tests/test_repair.py\n"
             "!python scripts/exp3_repair.py --quick --resume 2>&1 | grep -E 'ESTIMATED|chosen|done|Error|Traceback'\n"
             "!python scripts/exp4_real_and_univariate.py --quick --resume 2>&1 | grep -E 'done|Error|Traceback'"),
    ("md", "Run `v2` (current): R2 with 32 pairs/step, lr 0.01, 400 steps, pull towards 1; fine-tuning in 64-row "
           "micro-batches x 4 (the first run `main` stopped at the start of full FT, suspected OOM). "
           "A fine-tuning failure is now logged to log.txt and the rest continues."),
    ("code", "!python scripts/exp3_repair.py --resume --run-name v2 2>&1 | grep -v -E 'Loading weights|Writing model|UserWarning|FutureWarning|warnings.warn|it/s]'\n"
             "!grep -E 'FAILED|crashed|Traceback' $RESULTS_DIR/exp3/v2/log.txt; ls $RESULTS_DIR/exp3/v2"),
    ("code", "!python scripts/exp4_real_and_univariate.py --resume --run-name v2 --exp3-run v2 2>&1 | grep -v -E 'Loading weights|httpx|UserWarning|warnings.warn|it/s]'"),
    ("code", "!head -80 $RESULTS_DIR/reports/exp3_summary_v2.md; cat $RESULTS_DIR/reports/exp4_summary_v2.md"),
    ("md", "## Later stages (cells added as they are implemented)"),
]

nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
      "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
for kind, src in cells:
    c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
out = Path(__file__).resolve().parents[1] / "colab" / "run_all.ipynb"
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
