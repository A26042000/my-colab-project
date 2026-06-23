"""
build_notebook.py
Generates Phase0a_Colab.ipynb — a clean, well-organized Colab notebook.
Run: python build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ----------------------------------------------------------------------------
md("""# Phase 0a — Feasibility Probe (Colab)
### Does request-level compute budgeting beat static pruning?

This notebook runs the **cheapest experiment that can kill or green-light the whole PhD project**, in well under an hour on a free Colab GPU.

**The one question:** when we give each request a *different* amount of compute based on how hard it is, do we beat giving every request the *same* amount (static pruning)?

We answer it with three curves:

| Curve | Meaning |
|---|---|
| **Static** | Every request gets the same compute reduction. The baseline to beat. |
| **Oracle** | Each request gets a *hindsight-perfect* reduction. The **ceiling** — best case possible. |
| **Signal** | Each request's reduction is chosen from a **cheap early signal** only. The realistic, publishable result. |

**Two numbers we read off the figure:**
- **Oracle gap** = Oracle − Static. *Is there any headroom at all?* If ~0 → **stop**, no signal can help.
- **Realization gap** = Signal − Static. *Can the cheap signal capture it?* This is the result that matters.

> **Before you run:** make sure the runtime has a GPU. `Runtime → Change runtime type → Hardware accelerator → GPU (T4 is fine)`.
""")

# ----------------------------------------------------------------------------
md("""## 1. Setup

Install dependencies and pull in the `phase0` package (the experiment code, kept in separate well-documented modules so this notebook stays readable).
""")

code("""# Install pinned dependencies (quiet). Takes ~1-2 min on first run.
!pip -q install "transformers>=4.44" "datasets>=2.20" "accelerate>=0.33" \\
    "bitsandbytes>=0.43" "torch" "matplotlib" "numpy" "nbformat" 2>/dev/null
print("dependencies installed")
""")

code("""# Get the phase0 package.
# OPTION A (recommended): clone your repo, then cd into the folder that
# DIRECTLY contains phase0/.  If you re-run this cell, remove the old clone
# first to avoid nested copies.
import os, shutil
if os.path.isdir("my-colab-project"):
    shutil.rmtree("my-colab-project")  # clean slate on re-run
!git clone https://github.com/A26042000/my-colab-project.git
%cd my-colab-project/phase0_project

# Sanity: the current directory should directly contain a 'phase0' folder.
import os
assert os.path.isdir("phase0"), (
    "phase0/ not found in current dir. Run %pwd and %cd to the folder that "
    "directly contains phase0/."
)

import importlib, phase0
importlib.reload(phase0)
print("OK — phase0 package found at:", phase0.__file__)
""")

# ----------------------------------------------------------------------------
md("""## 2. Configuration (pre-registered)

Everything you might tune lives in `phase0/config.py`. We print it here so each run records exactly what it used. **Pre-registering** these — especially the quality threshold and the go/no-go conditions — *before* looking at results is how we keep ourselves honest.
""")

code("""from phase0.config import CONFIG
import json
print(json.dumps(CONFIG.__dict__, indent=2, default=str))
""")

# ----------------------------------------------------------------------------
md("""## 3. Load the model

We start with **one small instruct model** (Phase 0a). Only expand to more
families (Phase 0b) *after* this one shows an oracle gap. Small + greedy
decoding = fast and reproducible.
""")

code("""from phase0.model_runner import ModelRunner

runner = ModelRunner(
    CONFIG.model_name,
    CONFIG.torch_dtype,
    load_in_4bit=getattr(CONFIG, "load_in_4bit", False),
)
print("device:", runner.device)
print("num decoder layers:", len(runner.layers))
""")

# ----------------------------------------------------------------------------
md("""## 4. Load the workload (easy → hard)

The difficulty signal can only prove itself if requests genuinely differ in
difficulty, so we pull three tiers with **objective** task metrics:

- **easy** — multiple-choice science (accuracy)
- **medium** — short factual QA (exact match)
- **hard** — grade-school math, multi-step (exact match)
""")

code("""from phase0.data import load_all_suites

items = load_all_suites(CONFIG.n_items_per_suite, seed=CONFIG.seed)
from collections import Counter
print("total items:", len(items))
print("by suite:", Counter(it["suite"] for it in items))
print("\\nexample item:\\n", items[0]["prompt"][:300], "\\n---\\nanswer:", items[0]["answer"])
""")

# ----------------------------------------------------------------------------
md("""## 5. Run the sweep  ⏳ (the slow cell)

For every item we:
1. **Profile** it once at full compute — capturing the cheap early signal and
   the oracle channel-importance ranking (both free by-products of one pass).
2. **Generate** an answer at each reduction level and score it (0/1).

This is the only slow part. With ~80 items/suite and 8 reduction levels it is
a few thousand short generations — minutes on a T4. Lower `n_items_per_suite`
in the config for a quick smoke test first.
""")

code("""from phase0.experiment import run_sweep

sweep = run_sweep(
    runner,
    items,
    reduction_levels=CONFIG.reduction_levels,
    early_layer_index=CONFIG.early_layer_index,
    max_new_tokens=64,
    log_every=10,
)
print("sweep complete:", len(sweep["records"]), "records")
""")

# ----------------------------------------------------------------------------
md("""## 6. Build the three curves

- **Static**: average quality when every request gets the same reduction.
- **Oracle**: each request uses the most reduction it can tolerate (hindsight),
  capped to hit a target average — the honest **upper bound**.
- **Signal**: each request's reduction comes from the **early signal only**,
  no hindsight — the realistic policy.
""")

code("""from phase0.experiment import static_curve, oracle_curve, signal_curve

static = static_curve(sweep)
oracle = oracle_curve(sweep)
signal, signal_direction = signal_curve(sweep, signal_key="norm_last")

print("static :", [(round(c,2), round(q,2)) for c,q in static])
print("oracle :", [(round(c,2), round(q,2)) for c,q in oracle])
print("signal :", [(round(c,2), round(q,2)) for c,q in signal])
print("signal direction chosen:", signal_direction)
""")

# ----------------------------------------------------------------------------
md("""## 7. The decision figure

This plot **is** the go/no-go. Read it left-to-right (less compute → more
compute). If the **oracle** curve sits clearly above **static** in the
low-compute region, headroom exists. If **signal** also rises above static,
the cheap signal is capturing that headroom — the publishable result.
""")

code("""from phase0.report import plot_curves
import os

os.makedirs(CONFIG.results_dir, exist_ok=True)
plot_curves(
    static, oracle, signal,
    title=f"Phase 0a — {CONFIG.model_name.split('/')[-1]}",
    save_path=os.path.join(CONFIG.results_dir, "phase0a_curves.png"),
)
""")

# ----------------------------------------------------------------------------
md("""## 8. The verdict (against pre-registered thresholds)

We now compute the two savings at matched quality and apply the **pre-registered**
go conditions from the config. The verdict is one of:

- **GO** — headroom exists *and* the signal captures enough of it → expand to
  more families, then Gates B & C.
- **SOFT-GO** — headroom exists but the simple signal is too weak → try a tiny
  linear probe before committing.
- **NO-GO** — even the oracle barely beats static → stop or pivot, cheaply.
""")

code("""from phase0.experiment import compute_gaps
from phase0.report import verdict, print_verdict, save_results

gaps = compute_gaps(static, oracle, signal,
                    target_quality_drop=CONFIG.quality_drop_tolerance)
print("gaps:", json.dumps(gaps, indent=2, default=float))

v = verdict(gaps, CONFIG)
print_verdict(v)

# Persist everything for the write-up / reproducibility.
save_results(
    {
        "config": CONFIG.__dict__,
        "curves": {"static": static, "oracle": oracle, "signal": signal},
        "signal_direction": signal_direction,
        "gaps": gaps,
        "verdict": v,
    },
    CONFIG.results_dir,
)
""")

# ----------------------------------------------------------------------------
md("""## 9. What next?

- **GO** → re-run Section 3–8 with a second and third model family (edit
  `CONFIG.model_name`). Consistent gaps across families = strong evidence.
  Then move to **Gate B** (is the signal cheap to extract in a real datapath?)
  and **Gate C** (can a single-cycle scheduler realize the tolerance online?).
- **SOFT-GO** → swap the single signal for a tiny linear probe over several
  early statistics (`norm_mean/std/max/last` are already captured). If the
  probe clears the bar, proceed with the "near-free" framing.
- **NO-GO** → the abstraction lacks headroom on this setup. Before abandoning,
  sanity-check: different early layer? different operator (heads/layers)?
  harder workload with more difficulty spread? If still flat, pivot.

**Honesty reminders baked into this study:**
- The compute axis is a **proxy** (fraction of MLP retained), *not* measured
  energy. Real energy/latency comes only in the hardware phase. Never report
  proxy gains as silicon gains.
- The **oracle uses hindsight** and is an upper bound, never the system's
  result.
- Results on one small model may not generalize — state exactly what you ran.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
    "accelerator": "GPU",
}

with open("Phase0a_Colab.ipynb", "w") as f:
    nbf.write(nb, f)
print("wrote Phase0a_Colab.ipynb")
