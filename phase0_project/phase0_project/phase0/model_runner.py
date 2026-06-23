"""
model_runner.py
===============
Wraps a HuggingFace causal-LM with the two capabilities Phase 0a needs:

  (1) SKIP MLP CHANNELS at a chosen reduction level, ranked either by an
      ORACLE measure (post-hoc, the upper bound) or left full.
  (2) EXTRACT AN EARLY DIFFICULTY SIGNAL (cheap hidden-state statistics from
      an early layer) during a normal forward pass.

KEY IDEAS FOR A BEGINNER
------------------------
* A transformer is a stack of identical "layers". Each layer has an MLP block
  (a big feed-forward network) that does most of the arithmetic. Inside the
  MLP, the "intermediate" dimension has thousands of CHANNELS. Skipping the
  least-useful channels is our compute-saving lever.

* "Skip a channel" here = zero out that channel's intermediate activation so
  it contributes nothing. This is a faithful *proxy* for not computing it.
  (Real hardware would actually not compute it; for a feasibility study,
  zeroing measures the QUALITY effect, which is what we need first.)

* The ORACLE ranking decides WHICH channels to drop using information from
  the full run (mean absolute activation per channel). That's "cheating with
  hindsight" on purpose: it tells us the BEST CASE. The real signal-driven
  policy comes later and is NOT allowed this hindsight.

We implement skipping with forward HOOKS: small functions PyTorch calls each
time data flows through a module, letting us modify activations on the fly
WITHOUT editing the model's source.
"""

from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelRunner:
    def __init__(self, model_name: str, torch_dtype: str = "float16", load_in_4bit: bool = False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = getattr(torch, torch_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        load_kwargs = {"dtype": dtype}
        if load_in_4bit and self.device == "cuda":
            # 4-bit quantization lets a 7B model fit on a free T4 (16 GB).
            # Requires bitsandbytes (installed in the notebook).
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, **load_kwargs
            ).to(self.device)
        self.model.eval()

        # Locate the list of decoder layers in a model-agnostic way.
        # Most HF causal LMs expose them at model.model.layers.
        self.layers = self._find_decoder_layers()

        # State used by hooks (set per-run, then cleared).
        self._skip_fraction: float = 0.0
        self._channel_rank: Optional[Dict[int, torch.Tensor]] = None
        self._captured_early: Optional[torch.Tensor] = None
        self._hook_handles: List = []

    # ------------------------------------------------------------------ #
    # Setup helpers
    # ------------------------------------------------------------------ #
    def _find_decoder_layers(self):
        m = self.model
        for path in [("model", "layers"), ("transformer", "h"), ("gpt_neox", "layers")]:
            obj = m
            ok = True
            for attr in path:
                if hasattr(obj, attr):
                    obj = getattr(obj, attr)
                else:
                    ok = False
                    break
            if ok:
                return obj
        raise RuntimeError(
            "Could not locate decoder layers. Inspect `model` and adjust "
            "_find_decoder_layers for this architecture."
        )

    def _mlp_down_proj(self, layer):
        """
        Return the MLP submodule whose INPUT is the intermediate activation
        (the thing whose channels we zero). In Llama/Qwen this is `mlp.down_proj`;
        its input has the intermediate dimension we prune.
        """
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            return layer.mlp.down_proj
        # Fallback names for other architectures.
        for name in ["c_proj", "dense_4h_to_h", "fc2"]:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, name):
                return getattr(layer.mlp, name)
        raise RuntimeError("Could not find MLP down-projection for this model.")

    # ------------------------------------------------------------------ #
    # Pass 1: full run that also (a) records oracle channel importance and
    #         (b) captures the early difficulty signal.
    # ------------------------------------------------------------------ #
    def profile_request(self, prompt: str, early_layer_index: int) -> Dict:
        """
        Run the prompt ONCE at full compute. Return:
          - "early_signal": cheap stats from an early layer's hidden state
          - "channel_importance": per-layer mean |activation| (oracle ranking)
        Both are by-products of a single forward pass.
        """
        importance: Dict[int, torch.Tensor] = {}
        captured = {}

        handles = []

        # Hook on each MLP down_proj INPUT to measure channel importance.
        def make_importance_hook(idx):
            def hook(module, inputs):
                # inputs[0]: [batch, seq, intermediate_dim]
                act = inputs[0].detach()
                # Mean absolute value per channel = how active each channel is.
                imp = act.abs().mean(dim=(0, 1)).float().cpu()
                importance[idx] = imp
            return hook

        # Hook on the early layer OUTPUT to capture the difficulty signal.
        def early_hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["early"] = hs.detach()

        for idx, layer in enumerate(self.layers):
            dp = self._mlp_down_proj(layer)
            handles.append(dp.register_forward_pre_hook(make_importance_hook(idx)))
        handles.append(
            self.layers[early_layer_index].register_forward_hook(early_hook)
        )

        enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            self.model(**enc)

        for h in handles:
            h.remove()

        early_hs = captured["early"]  # [1, seq, hidden]
        signal = self._summarize_early(early_hs)
        return {"early_signal": signal, "channel_importance": importance}

    @staticmethod
    def _summarize_early(early_hs: torch.Tensor) -> Dict[str, float]:
        """
        Turn an early hidden-state tensor into a few CHEAP scalar features.
        These are the candidate 'difficulty signals'. They are cheap because
        they are simple statistics of activations the model already produced.
        """
        # Per-token L2 norm, then summarize across tokens.
        norms = early_hs[0].float().norm(dim=-1)  # [seq]
        return {
            "norm_mean": norms.mean().item(),
            "norm_std": norms.std().item(),
            "norm_max": norms.max().item(),
            "norm_last": norms[-1].item(),  # last token often summarizes prompt
        }

    # ------------------------------------------------------------------ #
    # Pass 2: generate an answer at a chosen reduction level.
    # ------------------------------------------------------------------ #
    def _install_skip_hooks(self, reduction: float, channel_rank: Dict[int, torch.Tensor]):
        """
        Install hooks that zero the LOWEST-importance channels at the MLP
        down_proj input, for the given reduction fraction.
        """
        self._remove_skip_hooks()
        if reduction <= 0.0:
            return

        def make_skip_hook(idx):
            rank = channel_rank.get(idx)
            def hook(module, inputs):
                if rank is None:
                    return None
                x = inputs[0]
                n = x.shape[-1]
                k = int(n * reduction)
                if k <= 0:
                    return None
                # Indices of the k least-important channels (drop these).
                drop_idx = torch.topk(rank, k, largest=False).indices.to(x.device)
                x = x.clone()
                x[..., drop_idx] = 0.0
                return (x,) + inputs[1:]
            return hook

        for idx, layer in enumerate(self.layers):
            dp = self._mlp_down_proj(layer)
            self._hook_handles.append(dp.register_forward_pre_hook(make_skip_hook(idx)))

    def _remove_skip_hooks(self):
        for h in self._hook_handles:
            h.remove()
        self._hook_handles = []

    def generate(
        self,
        prompt: str,
        reduction: float,
        channel_rank: Optional[Dict[int, torch.Tensor]],
        max_new_tokens: int = 64,
    ) -> str:
        """
        Generate an answer with `reduction` fraction of MLP channels skipped,
        dropping the least-important channels per `channel_rank`.
        reduction=0 (and channel_rank=None) gives the full-compute answer.
        """
        self._install_skip_hooks(reduction, channel_rank or {})
        try:
            enc = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,  # greedy = deterministic = reproducible
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            text = self.tokenizer.decode(
                out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True
            )
        finally:
            self._remove_skip_hooks()  # ALWAYS clean up, even on error
        return text
