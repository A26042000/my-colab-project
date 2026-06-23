"""
report.py
=========
Turns experiment numbers into (a) the decision figure and (b) an explicit
GO / SOFT-GO / NO-GO verdict against the PRE-REGISTERED thresholds.

This file is where Phase 0a "speaks": the figure becomes Figure 1 of the paper
if the project is green-lit, and the verdict tells you whether to proceed.
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_curves(
    static: List[Tuple[float, float]],
    oracle: List[Tuple[float, float]],
    signal: List[Tuple[float, float]],
    title: str,
    save_path: str = None,
):
    """Plot the three curves: quality vs. average compute used (proxy)."""
    plt.figure(figsize=(7, 5))
    for curve, label, style in [
        (static, "Static pruning (baseline)", "o-"),
        (oracle, "Oracle dynamic (ceiling)", "s--"),
        (signal, "Signal-driven (realistic)", "^-"),
    ]:
        xs = [p[0] for p in sorted(curve)]
        ys = [p[1] for p in sorted(curve)]
        plt.plot(xs, ys, style, label=label, linewidth=2, markersize=6)
    plt.xlabel("Average compute used (proxy: fraction of MLP retained)")
    plt.ylabel("Quality (task metric, 0..1)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_xaxis()  # left = less compute = the interesting region
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def verdict(gaps: Dict, cfg) -> Dict:
    """
    Apply the PRE-REGISTERED go conditions from config.py and return a verdict.

    Gate A logic:
      - oracle_saving >= min_oracle_compute_saving  -> headroom exists
      - signal_saving >= min_signal_capture_fraction * oracle_saving
                                                     -> signal captures enough
    """
    oracle_saving = gaps["oracle_saving"]
    signal_saving = gaps["signal_saving"]

    # Guard: if the sample was too small/noisy (flagged in compute_gaps) or the
    # model is too weak to do the task at all (very low full quality), we must
    # NOT issue a real verdict — the numbers are meaningless. Smoke tests land
    # here by design.
    too_noisy = gaps.get("noise_warning", False)
    too_weak = gaps.get("full_quality", 0.0) < 0.4  # <40% correct at full compute
    if too_noisy or too_weak:
        why = []
        if too_weak:
            why.append(
                f"model solves only {gaps.get('full_quality', 0):.0%} at full "
                f"compute (need >= 40% for a meaningful test — use a stronger model)"
            )
        if too_noisy:
            why.append("curves are noisy (too few items — raise n_items_per_suite)")
        return {
            "decision": "INCONCLUSIVE",
            "reason": "Not a real verdict: " + "; ".join(why) + ". "
            "Fix these and re-run before trusting any GO/NO-GO.",
            "oracle_saving": oracle_saving,
            "signal_saving": signal_saving,
            "capture_fraction": 0.0,
        }

    headroom_ok = oracle_saving >= cfg.min_oracle_compute_saving
    if oracle_saving > 0:
        capture_fraction = signal_saving / oracle_saving
    else:
        capture_fraction = 0.0
    capture_ok = capture_fraction >= cfg.min_signal_capture_fraction

    if not headroom_ok:
        decision = "NO-GO"
        reason = (
            f"Oracle saves only {oracle_saving:.1%} compute at matched quality "
            f"(need >= {cfg.min_oracle_compute_saving:.0%}). Even a perfect "
            f"hindsight policy barely beats static pruning -> request-level "
            f"budgeting has no headroom on this model/workload. STOP or pivot."
        )
    elif headroom_ok and not capture_ok:
        decision = "SOFT-GO"
        reason = (
            f"Headroom EXISTS (oracle saves {oracle_saving:.1%}) but the simple "
            f"signal captures only {capture_fraction:.0%} of it "
            f"(need >= {cfg.min_signal_capture_fraction:.0%}). Try a tiny linear "
            f"probe over several signals before committing; the premise is alive "
            f"but the cheapest signal is not yet sufficient."
        )
    else:
        decision = "GO"
        reason = (
            f"Headroom exists (oracle saves {oracle_saving:.1%}) AND the simple "
            f"signal captures {capture_fraction:.0%} of it (saving "
            f"{signal_saving:.1%} compute at matched quality). Evidence supports "
            f"the killer result. Expand to more model families, then Gates B & C."
        )

    return {
        "decision": decision,
        "reason": reason,
        "oracle_saving": oracle_saving,
        "signal_saving": signal_saving,
        "capture_fraction": capture_fraction,
    }


def save_results(payload: Dict, results_dir: str, name: str = "phase0a_results.json"):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, name)
    # Make numpy types JSON-serializable.
    def clean(o):
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        return o
    with open(path, "w") as f:
        json.dump(clean(payload), f, indent=2)
    print(f"Saved results -> {path}")
    return path


def print_verdict(v: Dict):
    bar = "=" * 70
    print(bar)
    print(f"  PHASE 0a VERDICT:  {v['decision']}")
    print(bar)
    print(v["reason"])
    print(bar)
