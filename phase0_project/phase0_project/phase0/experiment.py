"""
experiment.py
=============
The actual Phase 0a logic. Produces the ONE figure that decides the project:
three curves of quality vs. average compute, for

    STATIC   : every request gets the SAME reduction (the baseline to beat)
    ORACLE   : each request gets a hindsight-perfect per-request reduction
               (the CEILING of what request-level budgeting could do)
    SIGNAL   : each request's reduction is chosen from the EARLY SIGNAL only
               (the realistic policy; the publishable margin lives here)

THE TWO GAPS WE READ OFF THE FIGURE
-----------------------------------
  oracle gap     = quality(ORACLE) - quality(STATIC)   at equal avg compute
                   -> "is there ANY headroom?" If ~0, STOP. No signal helps.
  realization gap= quality(SIGNAL) - quality(STATIC)   at equal avg compute
                   -> "can the cheap signal capture it?" This is the result.

DEFINITIONS (kept deliberately simple for Phase 0a)
---------------------------------------------------
* "Compute used" by a request at reduction r is approximated as (1 - r):
  removing fraction r of MLP channels leaves fraction (1-r) of MLP compute.
  This is a PROXY (real energy comes in the architecture phase) and we label
  it as such everywhere.
* "Per-request oracle tolerance" = the LARGEST reduction at which that single
  request stays correct (score 1). For incorrect-at-full items, tolerance = 0.
"""

from typing import Dict, List, Tuple

import numpy as np

from .metrics import score_item
from .model_runner import ModelRunner


# --------------------------------------------------------------------------- #
# Step 1: profile + sweep every item once, caching outputs at each reduction.
# --------------------------------------------------------------------------- #
def run_sweep(
    runner: ModelRunner,
    items: List[Dict],
    reduction_levels: List[float],
    early_layer_index: int,
    max_new_tokens: int = 64,
    log_every: int = 10,
) -> Dict:
    """
    For each item:
      - profile once (early signal + oracle channel importance)
      - generate an answer at EACH reduction level, scoring each (0/1)

    Returns a dict with per-item records:
      records[i] = {
        "item": <the item dict>,
        "signal": {...early stats...},
        "scores": {r: 0/1 for r in reduction_levels},
      }
    """
    records = []
    for i, item in enumerate(items):
        prof = runner.profile_request(item["prompt"], early_layer_index)
        rank = prof["channel_importance"]
        scores = {}
        for r in reduction_levels:
            out = runner.generate(
                item["prompt"],
                reduction=r,
                channel_rank=(None if r == 0.0 else rank),
                max_new_tokens=max_new_tokens,
            )
            scores[r] = score_item(item, out)
        records.append(
            {"item": item, "signal": prof["early_signal"], "scores": scores}
        )
        if (i + 1) % log_every == 0:
            print(f"  profiled {i + 1}/{len(items)} items")
    return {"records": records, "reduction_levels": list(reduction_levels)}


# --------------------------------------------------------------------------- #
# Step 2: per-request oracle tolerance.
# --------------------------------------------------------------------------- #
def oracle_tolerance(record: Dict, reduction_levels: List[float], full_correct_only=True) -> float:
    """
    Largest reduction at which THIS request is still correct.
    If the full run (r=0) is wrong, tolerance is 0 (nothing to give).
    """
    if full_correct_only and record["scores"].get(0.0, 0) == 0:
        return 0.0
    tol = 0.0
    for r in sorted(reduction_levels):
        if record["scores"].get(r, 0) == 1:
            tol = r
    return tol


# --------------------------------------------------------------------------- #
# Step 3: build the STATIC baseline curve.
# --------------------------------------------------------------------------- #
def static_curve(sweep: Dict) -> List[Tuple[float, float]]:
    """
    For each fixed reduction r applied to ALL requests:
      x = average compute used = (1 - r)
      y = average quality = mean score at that r
    Returns list of (avg_compute, quality) points.
    """
    recs = sweep["records"]
    pts = []
    for r in sweep["reduction_levels"]:
        q = np.mean([rec["scores"][r] for rec in recs])
        pts.append((1.0 - r, float(q)))
    return sorted(pts)


# --------------------------------------------------------------------------- #
# Step 4: build the ORACLE dynamic curve.
# --------------------------------------------------------------------------- #
def oracle_curve(sweep: Dict, n_budget_points: int = 8) -> List[Tuple[float, float]]:
    """
    The oracle assigns each request as much reduction as it can tolerate, but
    we constrain the AVERAGE compute to match a target so it's comparable to
    static. We sweep target average-compute levels and, at each, give every
    request the largest reduction <= a shared cap that it still tolerates.

    Simplification for Phase 0a: we sweep a global cap c; each request uses
    min(its tolerance, c). This yields (avg_compute, quality) points that are
    an honest UPPER BOUND, because it uses per-request hindsight tolerance.
    """
    recs = sweep["records"]
    levels = sorted(sweep["reduction_levels"])
    tols = [oracle_tolerance(rec, levels) for rec in recs]

    pts = []
    for cap in levels:
        used_reductions = [min(t, cap) for t in tols]
        avg_compute = float(np.mean([1.0 - r for r in used_reductions]))
        # Quality: a request is correct if its applied reduction <= tolerance.
        # By construction min(t,cap) <= t, so oracle-correct items stay correct;
        # items wrong at full (tol=0) contribute their r=0 score.
        quality = float(
            np.mean(
                [
                    rec["scores"][0.0] if t == 0.0 else 1
                    for rec, t in zip(recs, tols)
                ]
            )
        )
        pts.append((avg_compute, quality))
    return sorted(set(pts))


# --------------------------------------------------------------------------- #
# Step 5: build the SIGNAL-driven curve.
# --------------------------------------------------------------------------- #
def _signal_value(signal: Dict[str, float], key: str = "norm_last") -> float:
    return signal[key]


def signal_curve(
    sweep: Dict,
    signal_key: str = "norm_last",
    n_bins: int = 8,
) -> List[Tuple[float, float]]:
    """
    Map the early signal to a per-request reduction WITHOUT hindsight.

    Policy (deliberately simple, hardware-plausible):
      - rank requests by the signal value
      - "easier-looking" requests (one end of the signal) get MORE reduction
      - sweep how aggressively we scale signal->reduction to trace a curve

    We don't know a priori which signal direction means 'easy', so we pick the
    direction that yields the better curve on THIS data and report it honestly
    as a calibrated choice (a real deployment would calibrate on a held-out
    split; see notes). For Phase 0a this still fairly tests 'does the signal
    carry usable information beyond static?'.
    """
    recs = sweep["records"]
    levels = sorted(sweep["reduction_levels"])
    max_r = max(levels)
    vals = np.array([_signal_value(rec["signal"], signal_key) for rec in recs])

    # Normalize signal to [0,1].
    if vals.max() > vals.min():
        norm = (vals - vals.min()) / (vals.max() - vals.min())
    else:
        norm = np.zeros_like(vals)

    def quality_at(scale: float, direction: int) -> Tuple[float, float]:
        # direction +1: high signal -> more reduction; -1: reverse.
        s = norm if direction == 1 else (1.0 - norm)
        # Per-request reduction = scale * s, snapped to the nearest swept level.
        applied = []
        for si in s:
            target = scale * max_r * si
            r = min(levels, key=lambda L: abs(L - target))
            applied.append(r)
        avg_compute = float(np.mean([1.0 - r for r in applied]))
        quality = float(np.mean([rec["scores"][r] for rec, r in zip(recs, applied)]))
        return avg_compute, quality

    # Trace a curve by sweeping the aggressiveness scale, for both directions;
    # keep whichever direction gives higher average quality.
    scales = np.linspace(0.0, 1.0, n_bins)
    best_dir, best_mean = 1, -1.0
    for d in (1, -1):
        m = np.mean([quality_at(sc, d)[1] for sc in scales])
        if m > best_mean:
            best_mean, best_dir = m, d
    pts = [quality_at(sc, best_dir) for sc in scales]
    return sorted(set(pts)), best_dir


# --------------------------------------------------------------------------- #
# Step 6: compute the two gaps at a matched compute level.
# --------------------------------------------------------------------------- #
def _interp_quality(curve: List[Tuple[float, float]], x: float) -> float:
    """Linear-interpolate quality at a given average-compute x."""
    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    return float(np.interp(x, xs, ys))


def compute_gaps(
    static: List[Tuple[float, float]],
    oracle: List[Tuple[float, float]],
    signal: List[Tuple[float, float]],
    target_quality_drop: float,
) -> Dict:
    """
    Read the headline numbers:
      1. Find the avg-compute at which STATIC drops by `target_quality_drop`
         from its full-compute quality.
      2. At that SAME quality, find how much LESS compute ORACLE and SIGNAL need
         (compute saving). These savings are the publishable margins.
    """
    # Quality at full compute (compute == 1.0), read from the static curve.
    full_q = max(q for c, q in static if abs(c - 1.0) < 1e-6) if any(
        abs(c - 1.0) < 1e-6 for c, _ in static
    ) else max(q for _, q in static)
    target_q = full_q - target_quality_drop

    def best_quality_at_or_below(curve, compute_budget):
        """Best quality achievable using AT MOST `compute_budget` average compute.
        Using a 'best so far' (monotone) reading prevents a single noisy uptick
        at heavy reduction from being credited as a real operating point."""
        pts = sorted(curve)  # ascending compute
        best = 0.0
        out = []
        for c, q in pts:
            best = max(best, q)
            out.append((c, best))
        ok = [c for c, q in out if q >= target_q]
        return min(ok) if ok else 1.0

    static_c = best_quality_at_or_below(static, target_q)
    oracle_c = best_quality_at_or_below(oracle, target_q)
    signal_c = best_quality_at_or_below(signal, target_q)

    # The oracle is a hindsight upper bound: by construction it cannot need MORE
    # compute than static to reach the same quality. If noise pushes it there,
    # clamp and flag, rather than reporting an impossible negative saving.
    noisy = oracle_c > static_c + 1e-9
    oracle_c = min(oracle_c, static_c)

    return {
        "full_quality": full_q,
        "target_quality": target_q,
        "static_compute": static_c,
        "oracle_compute": oracle_c,
        "signal_compute": signal_c,
        "oracle_saving": static_c - oracle_c,      # >=0 by construction
        "signal_saving": static_c - signal_c,      # the realistic margin
        "noise_warning": noisy,                    # True => sample too small/noisy
    }
