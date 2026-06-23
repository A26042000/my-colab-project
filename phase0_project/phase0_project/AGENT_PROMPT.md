# Agent Prompt — Phase 0a Implementation & Iteration

Copy everything below the line into Claude Code or Codex as your opening instruction. It gives the agent the full context, the guardrails that matter, and a precise task list. Keep the four project documents (proposal_v2, phase0_plan, gateA_experiment_spec, glossary) in the repo so the agent can read them.

---

## ROLE

You are helping implement **Phase 0a** of a PhD research project in computer architecture. Phase 0a is a *feasibility probe*: a cheap software experiment that decides whether the whole project is worth pursuing. You are not building the final hardware or the final system — you are building the experiment that produces a single go/no-go figure.

## THE PROJECT IN ONE PARAGRAPH

The project proposes **request-level compute budgeting as an architectural primitive for LLM inference**: a hardware controller reads a cheap signal early in the model's forward pass, estimates how much compute the current request needs, sets a per-request budget, and skips non-essential work (starting with MLP channels) to stay within it — no fine-tuning. Phase 0a tests the make-or-break premise: **does spending a per-request compute budget where it matters beat spending the same average compute uniformly (static pruning)?**

## WHAT ALREADY EXISTS IN THIS REPO

A working first implementation:

```
phase0/
  __init__.py
  config.py         # all pre-registered settings (ONE place to tune)
  data.py           # loads easy/medium/hard suites into a uniform shape
  metrics.py        # real task-metric scoring (NOT self-similarity)
  model_runner.py   # model wrapper: MLP-channel skipping + early-signal capture
  experiment.py     # builds static/oracle/signal curves + computes the two gaps
  report.py         # plots the decision figure + GO/SOFT-GO/NO-GO verdict
build_notebook.py   # regenerates Phase0a_Colab.ipynb
Phase0a_Colab.ipynb # the Colab notebook (run top to bottom)
```

Read `gateA_experiment_spec.md` and `phase0_plan.md` first — they define exactly what the experiment must measure and why.

## NON-NEGOTIABLE GUARDRAILS (these encode hard-won reviewer feedback)

1. **The oracle is an upper bound, never the result.** The oracle ranks channels using post-hoc (full-run) knowledge the real hardware will not have. Report oracle numbers only as a ceiling. Never present oracle gains as the system's performance.
2. **Score with real task metrics, not similarity to the full-compute output.** The full model is sometimes wrong; a reduced model can be different but still correct. Self-similarity is at most a secondary metric.
3. **The compute axis is a PROXY** (fraction of MLP retained), not measured energy. Label it as a proxy everywhere. Real energy/latency belongs to a later hardware phase.
4. **Pre-register thresholds before looking at results.** The quality threshold and go/no-go conditions live in `config.py`. Do not tune them to make results look good after the fact.
5. **Measure decision safety, not just correlation.** A signal that correlates on average but starves hard requests is unsafe. Track the false-low-budget rate (hard requests given too little compute) when you extend the metrics.
6. **Report prefill and decode separately** when you add timing/compute breakdowns.
7. **Keep it cheap and reproducible.** Greedy decoding, fixed seed, small item counts for Phase 0a. The whole notebook must run on a free Colab T4 in well under an hour.

## YOUR TASKS (in order)

**Task 1 — Verify it runs end to end.** Execute the notebook (or an equivalent local script) on the smallest possible setting (`n_items_per_suite = 8`). Fix any breakage from model-architecture differences in `model_runner.py` (the hook target names may differ across model families — make `_find_decoder_layers` and `_mlp_down_proj` robust and add a clear error if a new architecture is unsupported).

**Task 2 — Sanity-check the signal capture.** The current signal policy in `experiment.py:signal_curve` picks the better of two directions on the same data it evaluates. That is fine for a first look but is mildly optimistic. Add a proper **train/test split**: calibrate the signal→reduction direction and scale on a held-out calibration split, then report the curve on the evaluation split. Keep the old behavior available behind a flag for comparison.

**Task 3 — Add the decision-safety metrics.** Implement, in a new `phase0/safety.py`, the budget-classification accuracy and the false-low-budget rate described in `gateA_experiment_spec.md` Section 6, and surface them in `report.py`'s verdict output.

**Task 4 — Add the tiny linear probe (the SOFT-GO path).** When a single raw signal is too weak, fit a small linear model over the already-captured early statistics (`norm_mean/std/max/last`) to predict per-request tolerance, on the calibration split only. Add it as an alternative to the single-signal policy and report whether it clears the bar.

**Task 5 — Multi-model loop.** Make it trivial to run Tasks 1–4 across 2–3 model families and produce a combined summary table (one row per model: oracle saving, signal saving, capture fraction, verdict). This is the cross-family evidence the proposal needs.

## STYLE REQUIREMENTS

- Every new function gets a short docstring explaining *what* and *why*, in plain language a beginner can follow — match the existing files' tone.
- Prefer small, readable, well-named functions over clever one-liners.
- No silent failures: if a dataset or model can't load, raise a clear, actionable error.
- Keep all tunables in `config.py`. Do not scatter magic numbers.
- After each task, print a one-paragraph summary of what changed and what the agent observed when running it.

## DEFINITION OF DONE FOR PHASE 0a

A reproducible run that, for each model, outputs: the three-curve figure, the two gaps, the safety metrics, and a GO / SOFT-GO / NO-GO verdict against the pre-registered thresholds — plus a short written interpretation that an advisor could read in two minutes.
