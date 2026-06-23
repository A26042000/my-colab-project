# Phase 0a — Feasibility Probe

This repo runs the **cheapest experiment that can decide a PhD project**: does giving each LLM request a *different* amount of compute based on its difficulty beat giving every request the *same* amount (static pruning)?

It produces one figure (three curves) and one verdict (GO / SOFT-GO / NO-GO).

## Quick start (Google Colab — recommended)

1. Open `Phase0a_Colab.ipynb` in Colab.
2. `Runtime → Change runtime type → GPU` (a free T4 is enough).
3. Make the `phase0/` folder available to the notebook — either:
   - **clone**: put `!git clone <your-repo>` + `%cd <repo>` in Section 1, or
   - **upload**: drag the `phase0/` folder into Colab's Files panel so it sits next to the notebook.
4. Run cells top to bottom. The whole thing finishes in well under an hour.

## Quick start (local)

```bash
pip install -r requirements.txt
python build_notebook.py          # regenerate the notebook if you edit it
# or drive the package directly:
python -c "
from phase0.config import CONFIG
from phase0.model_runner import ModelRunner
from phase0.data import load_all_suites
from phase0.experiment import run_sweep, static_curve, oracle_curve, signal_curve, compute_gaps
from phase0.report import plot_curves, verdict, print_verdict
CONFIG.n_items_per_suite = 8          # smoke test
r = ModelRunner(CONFIG.model_name, CONFIG.torch_dtype)
items = load_all_suites(CONFIG.n_items_per_suite, CONFIG.seed)
s = run_sweep(r, items, CONFIG.reduction_levels, CONFIG.early_layer_index)
stat, orac = static_curve(s), oracle_curve(s)
sig, _ = signal_curve(s)
g = compute_gaps(stat, orac, sig, CONFIG.quality_drop_tolerance)
print_verdict(verdict(g, CONFIG))
"
```

## What each file does

| File | Role |
|---|---|
| `phase0/config.py` | Every tunable setting, pre-registered in one place. |
| `phase0/data.py` | Loads easy / medium / hard workloads into one uniform shape. |
| `phase0/metrics.py` | Scores answers with **real task metrics** (not self-similarity). |
| `phase0/model_runner.py` | Skips MLP channels + captures the cheap early signal via forward hooks. |
| `phase0/experiment.py` | Builds the static / oracle / signal curves; computes the two gaps. |
| `phase0/report.py` | Plots the decision figure; emits the GO / SOFT-GO / NO-GO verdict. |
| `Phase0a_Colab.ipynb` | The notebook tying it together, with explanations. |
| `AGENT_PROMPT.md` | Drop-in instructions for Claude Code / Codex to extend this. |

## How to read the result

- **Oracle gap** (Oracle − Static): *is there any headroom?* If ~0, no signal can ever help → **NO-GO**.
- **Realization gap** (Signal − Static): *can the cheap signal capture it?* This is the publishable margin.

## Honesty notes (built into the design)

- The compute axis is a **proxy** (fraction of MLP retained), not measured energy.
- The **oracle uses hindsight** — it's a ceiling, never the system's result.
- One small model may not generalize; state exactly what you ran.

See `AGENT_PROMPT.md` to have an AI coding agent verify, harden, and extend this (train/test split, safety metrics, linear-probe SOFT-GO path, multi-model loop).
