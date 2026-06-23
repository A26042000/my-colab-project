"""
config.py
=========
Central configuration for the Phase 0a feasibility experiment.

WHY THIS FILE EXISTS
--------------------
Every knob you might want to change lives here, in ONE place. This keeps the
notebook clean and makes your runs reproducible: change a value here, re-run,
and you know exactly what differed. Pre-registering these values (writing them
down BEFORE looking at results) is how we avoid fooling ourselves.

Beginner note: a "dataclass" is just a tidy container for named settings.
You read/modify the values; you don't need to understand the machinery.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Phase0Config:
    # ---- Model ----------------------------------------------------------
    # We start with ONE small-but-capable open model (Phase 0a). It must be
    # strong enough to actually DO the tasks at full compute — otherwise there
    # is no degradation to measure. A 1.5B model is too weak; 7B is the sweet
    # spot on a free T4 (use 4-bit if memory is tight; see notebook).
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # Load in 4-bit to fit a 7B model on a free Colab T4 (16 GB). Set False if
    # you have more memory or use a smaller model.
    load_in_4bit: bool = True

    # Use float16 on GPU to fit memory and run fast. (bfloat16 also fine.)
    torch_dtype: str = "float16"

    # ---- Workload -------------------------------------------------------
    # We want requests spanning GENUINELY EASY -> GENUINELY HARD so the
    # difficulty signal has range to act on. Each entry: (name, n_items).
    # Keep n_items small for Phase 0a; raise later once the pipeline works.
    n_items_per_suite: int = 80

    # ---- The skippable operator ----------------------------------------
    # Phase 0a controls MLP channels (the model's bulk compute). We "remove"
    # a channel by zeroing its contribution. Reduction levels are the
    # FRACTION of channels removed; we sweep these to build curves.
    reduction_levels: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )

    # ---- Early difficulty signal ---------------------------------------
    # Which transformer layer counts as "early" for reading the signal.
    # The whole premise: a signal here predicts how much we can skip later.
    early_layer_index: int = 2

    # ---- Pre-registered quality threshold ------------------------------
    # A request "tolerates" a reduction if its task score stays within this
    # margin of the full-compute score. DECIDE THIS BEFORE RUNNING.
    # For accuracy/exact-match (0 or 1 per item) we operate at the dataset
    # level; see metrics.py for how the threshold is applied.
    quality_drop_tolerance: float = 0.05  # allow up to 5% absolute drop

    # ---- Pre-registered GO conditions ----------------------------------
    # Gate A (Phase 0a) go/no-go, written down in advance:
    # The oracle must reach the static baseline's quality using at least this
    # much LESS average compute. Below this, the abstraction has no headroom.
    min_oracle_compute_saving: float = 0.15  # 15% less average compute

    # The signal must capture at least this fraction of the oracle's gap.
    min_signal_capture_fraction: float = 0.5  # half of the oracle headroom

    # ---- Reproducibility ------------------------------------------------
    seed: int = 0

    # ---- Output ---------------------------------------------------------
    results_dir: str = "results"


# A single shared instance the notebook imports.
CONFIG = Phase0Config()
