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
             "!pip install -q -r requirements.txt && pip install -q -e . --no-deps\n"
             "# Colab ships torchao 0.10, which peft 0.21 rejects when it builds LoRA layers; nothing here needs torchao\n"
             "!pip uninstall -y -q torchao"),
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
    ("md", "LoRA for run `v2` (it failed there on Colab's torchao 0.10; fixed in cell 3). This trains LoRA only "
           "(full FT and the other methods are reused), adds it to the 152 finished cells with the same samples, "
           "then re-runs Exp4 as `v2lora` (Exp4 is only a few minutes)."),
    ("code", "!python scripts/exp3_repair.py --resume --run-name v2 --add-methods ft_lora 2>&1 | grep -v -E 'Loading weights|Writing model|UserWarning|FutureWarning|warnings.warn|it/s]'\n"
             "!grep -E 'FAILED|crashed|Traceback|added' $RESULTS_DIR/exp3/v2/log.txt | tail -5"),
    ("code", "!python scripts/exp4_real_and_univariate.py --resume --run-name v2lora --exp3-run v2 2>&1 | grep -v -E 'Loading weights|httpx|UserWarning|warnings.warn|it/s]'\n"
             "!head -40 $RESULTS_DIR/reports/exp3_summary_v2.md; cat $RESULTS_DIR/reports/exp4_summary_v2lora.md"),
    ("md", "## Stage 3 - Exp5 (ablations), figures, tables\n"
           "Exp5 needs the Exp3 run `v2` (R1 ranking, k/gamma, R2 gates). About 20-30 min on L4 (rough)."),
    ("code", "!python scripts/exp5_ablations.py --resume --exp3-run v2 2>&1 | grep -v -E 'Loading weights|UserWarning|warnings.warn|it/s]'\n"
             "!grep -E 'crashed|Traceback' $RESULTS_DIR/exp5/main/log.txt | tail -3"),
    ("code", "!python scripts/make_figures.py --exp3-run v2 --exp5-run main && "
             "python scripts/make_tables.py --exp3-run v2 --exp4-run v2lora --exp5-run main\n"
             "!ls $RESULTS_DIR/reports/figs/fig* $RESULTS_DIR/reports/tables/table?_*.tex; "
             "head -60 $RESULTS_DIR/reports/exp5_summary_main.md"),
    ("md", "## Stage 4 - real-world evidence, seeds, other TSFMs\n"
           "Run in this order (most important first); each is resumable and independent. Rough L4 times: Exp6 1-2 h "
           "(28 fev-bench covariate tasks + semi-synthetic EPF), Exp8 ~30-60 min (Chronos-2, TimesFM 3.0, "
           "TimesFM 2.5 + XReg), Exp7 ~1 h (R2 x 3 training seeds, test x 3 sample seeds). A100 is much faster."),
    ("code", "!python scripts/exp6_real_world.py --resume --exp3-run v2 2>&1 | grep -v -E 'Loading weights|httpx|UserWarning|warnings.warn|it/s]'\n"
             "!grep -E 'FAILED|crashed' $RESULTS_DIR/exp6/main/log.txt | tail -5; head -50 $RESULTS_DIR/reports/exp6_summary_main.md"),
    ("code", "!python scripts/exp8_other_models.py --quick 2>&1 | grep -E '\\| g |FAILED|skipped|Error|Traceback' | tail -8\n"
             "!python scripts/exp8_other_models.py --resume 2>&1 | grep -v -E 'Loading weights|httpx|UserWarning|warnings.warn|it/s]'\n"
             "!grep -E 'FAILED|crashed|skipped' $RESULTS_DIR/exp8/main/log.txt | tail -5; cat $RESULTS_DIR/reports/exp8_summary_main.md"),
    ("code", "!python scripts/exp7_seeds.py --resume --exp3-run v2 2>&1 | grep -v -E 'Loading weights|UserWarning|warnings.warn|it/s]'\n"
             "!grep -E 'crashed' $RESULTS_DIR/exp7/main/log.txt | tail -3; head -30 $RESULTS_DIR/reports/exp7_summary_main.md"),
    ("md", "## Stage 5 - multisine diagnosis, semi-synthetic on all fev tasks, cost\n"
           "Re-run cell 2 first to pull the new scripts. Rough A100 times: Exp9 ~20-30 min (190 cells), Exp10 ~30-60 min "
           "(28 tasks x 3 systems x 2 effect sizes, incl. data download and the fine-tuned models), Exp11 ~5 min."),
    ("code", "!python scripts/exp9_multisine.py --quick --resume 2>&1 | grep -E 'cells to run|summary|Error|Traceback'\n"
             "!python scripts/exp9_multisine.py --resume --exp3-run v2 2>&1 | grep -v -E 'Loading weights|httpx|UserWarning|warnings.warn|it/s]'\n"
             "!grep -E 'FAILED|crashed' $RESULTS_DIR/exp9/main/log.txt | tail -5; cat $RESULTS_DIR/reports/exp9_summary_main.md"),
    ("code", "!python scripts/exp10_semisynth_fev.py --resume --exp3-run v2 2>&1 | grep -v -E 'Loading weights|httpx|UserWarning|warnings.warn|it/s]|symlinks'\n"
             "!grep -E 'FAILED|crashed|skipped' $RESULTS_DIR/exp10/main/log.txt | tail -8; head -60 $RESULTS_DIR/reports/exp10_summary_main.md"),
    ("code", "!python scripts/exp11_cost.py --exp3-run v2 2>&1 | grep -E 'exp11:|Error|Traceback'\n"
             "!cat $RESULTS_DIR/reports/exp11_summary_main.md"),
    ("md", "## Stage 6 - Low Carbon London dToU pricing trial: data and model-free ground truth\n"
           "No GPU needed (a CPU runtime is fine; cells 1-3 and 5 are enough). Re-run cell 2 first. Downloads the "
           "London Datastore files (CC BY 4.0, ~800 MB zip) to local disk, builds the household x half-hour matrix "
           "(~1 GB, copied to Drive under `pakdd27/data/lcl/processed`), then estimates the price effects with the "
           "Std group as control (`$RESULTS_DIR/lcl/feasibility`). Rough time: download 5-15 min, prepare 15-30 min, "
           "feasibility < 5 min."),
    ("code", "LCL_ZIP = 'https://data.london.gov.uk/download/vqm0d/04feba67-f1a3-4563-98d0-f3071e3d56d1/Partitioned%20LCL%20Data.zip'\n"
             "LCL_TAR = 'https://data.london.gov.uk/download/vqm0d/14855047-44c2-4856-8a48-e5649200e6ce/Tariffs.xlsx'\n"
             "!mkdir -p /content/lcl/raw\n"
             "!test -s /content/lcl/raw/Partitioned_LCL_Data.zip || wget -q --show-progress -O /content/lcl/raw/Partitioned_LCL_Data.zip \"{LCL_ZIP}\"\n"
             "!test -s /content/lcl/raw/Tariffs.xlsx || wget -q -O /content/lcl/raw/Tariffs.xlsx \"{LCL_TAR}\"\n"
             "# expected sizes (London Datastore metadata): zip 795722689 bytes, xlsx 245384 bytes (md5 d64694233f1e3b7241fb1db02ff063f2)\n"
             "!ls -l /content/lcl/raw && md5sum /content/lcl/raw/Tariffs.xlsx"),
    ("code", "# prints the zip layout, the first file's columns and a time-parse check before processing\n"
             "!rm -f /content/drive/MyDrive/pakdd27/data/lcl/processed/hh_kwh.tmp.npy\n"
             "!python scripts/lcl_prepare.py --raw /content/lcl/raw --out /content/lcl/processed > /content/lcl/prepare.log 2>&1; "
             "head -20 /content/lcl/prepare.log; echo ...; tail -40 /content/lcl/prepare.log\n"
             "!mkdir -p /content/drive/MyDrive/pakdd27/data/lcl && cp /content/lcl/prepare.log /content/drive/MyDrive/pakdd27/data/lcl/\n"
             "# copy the data to Drive only if preparation finished\n"
             "!test -s /content/lcl/processed/prepare_log.json && "
             "cp -r /content/lcl/processed /content/drive/MyDrive/pakdd27/data/lcl/ && "
             "ls -l /content/drive/MyDrive/pakdd27/data/lcl/processed || echo 'PREPARE FAILED - nothing copied'"),
    ("code", "!python scripts/lcl_feasibility.py --data /content/lcl/processed > /content/lcl/feasibility.log 2>&1; "
             "tail -60 /content/lcl/feasibility.log\n"
             "!mkdir -p $RESULTS_DIR/lcl/feasibility && cp /content/lcl/feasibility.log $RESULTS_DIR/lcl/feasibility/ && "
             "ls -l $RESULTS_DIR/lcl/feasibility"),
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
