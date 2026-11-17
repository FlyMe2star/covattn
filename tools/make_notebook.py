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
           "The canonical Exp1 run was done locally (`exp1/main`). Optional: upload `exp1_results.zip` to "
           "`MyDrive/pakdd27/` and unpack it here so later stages / figures can read it."),
    ("code", "!test -f /content/drive/MyDrive/pakdd27/exp1_results.zip && unzip -q -n "
             "/content/drive/MyDrive/pakdd27/exp1_results.zip -d $RESULTS_DIR && echo unpacked || echo 'no zip uploaded'"),
    ("code", "# re-run / complete Exp1 on this GPU (resumable). About 2-3 min model + ~10 min ARX on Colab CPU\n"
             "# !python scripts/exp1_phenomenon.py --resume --batch-size 256"),
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
