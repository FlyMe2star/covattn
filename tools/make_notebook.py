"""Regenerate colab/run_all.ipynb (keeps the notebook diff-able as code)."""
import json
from pathlib import Path

cells = [
    ("md", "# covattn - run all experiments on Colab (A100)\nEvery experiment cell uses `--resume`; re-run any cell after a disconnect."),
    ("code", "from google.colab import drive\ndrive.mount('/content/drive')"),
    ("code", "# Option A: clone the (anonymous) repo\n# !git clone <ANON_REPO_URL> /content/covattn\n"
             "# Option B: unzip a copy uploaded to Drive\n"
             "!mkdir -p /content/covattn && cd /content && unzip -q -o /content/drive/MyDrive/pakdd27/covattn.zip -d /content/\n"
             "%cd /content/covattn"),
    ("code", "!pip install -q -r requirements.txt && pip install -q -e ."),
    ("code", "!nvidia-smi\nimport torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"),
    ("code", "import os\nos.environ['RESULTS_DIR'] = '/content/drive/MyDrive/pakdd27/results'\nos.makedirs(os.environ['RESULTS_DIR'], exist_ok=True)"),
    ("md", "## Stage 0 - Exp1 (phenomenon)"),
    ("code", "!python scripts/verify_model.py"),
    ("code", "!python scripts/exp1_phenomenon.py --quick --resume"),
    ("code", "# full grid (19 systems x 4 inputs x L in {256,512,1024,2048}); resumable\n"
             "!python scripts/exp1_phenomenon.py --resume --batch-size 256"),
    ("code", "!head -60 $RESULTS_DIR/reports/exp1_summary_main.md"),
    ("md", "## Later stages (added as they are implemented)"),
]
nb = {"cells": [], "metadata": {"accelerator": "GPU", "colab": {"gpuType": "A100"},
       "kernelspec": {"name": "python3", "display_name": "Python 3"}}, "nbformat": 4, "nbformat_minor": 5}
for kind, src in cells:
    c = {"cell_type": "markdown" if kind == "md" else "code", "metadata": {}, "source": src}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    nb["cells"].append(c)
out = Path(__file__).resolve().parents[1] / "colab" / "run_all.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(out)
