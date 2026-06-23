"""
phase0 — feasibility-study package for the LLM compute-budgeting project.

Modules:
  config        : all pre-registered settings (one place to change knobs)
  data          : loads easy/medium/hard workload suites into a uniform shape
  metrics       : real task-metric scoring (no self-similarity shortcut)
  model_runner  : model wrapper with MLP-channel skipping + early-signal capture
  experiment    : builds static/oracle/signal curves and computes the two gaps
  report        : plots the decision figure and emits GO/SOFT-GO/NO-GO verdict
"""

from .config import CONFIG, Phase0Config  # noqa: F401
