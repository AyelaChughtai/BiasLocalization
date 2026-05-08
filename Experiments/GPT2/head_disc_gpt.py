"""
GPT-2: Head Re-Discovery on Discovery Set

Full 144-head ablation scan using ONLY discovery-set prompts (200 prompts).
Tests whether L10H9 still emerges as top separable head on a principled
discovery split, separate from dev/test.

For each head: 4 bias metrics. For top-10: full capability evaluation.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from functools import partial

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval_utils import (
    get_prompts, eval_bias, full_eval, bootstrap_ci,
    print_results, results_to_json, eval_wikitext_ppl
)

RESULTS_DIR = Path(__file__).parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("(GPT-2): HEAD SCAN ON DISCOVERY SET")
print("=" * 70)

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("gpt2", device=device)

discovery_prompts = get_prompts("discovery")
print("Discovery prompts: %d" % len(discovery_prompts))


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


# ═══════════════════════════════════════════════════════════════
# BASELINE
# ═══════════════════════════════════════════════════════════════
print("\nComputing baseline...")
baseline = eval_bias(model, discovery_prompts)
print("  Baseline signed bias: %+.4f" % baseline["signed_bias"])
print("  Baseline abs bias:     %.4f" % baseline["abs_bias"])

baseline_ppl_arr = eval_wikitext_ppl(model, hooks=None, n_sentences=50)
baseline_ppl = float(np.mean(baseline_ppl_arr))
print("  Baseline WikiText PPL: %.2f" % baseline_ppl)

# ═══════════════════════════════════════════════════════════════
# FULL 144-HEAD SCAN (bias metrics only — fast)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("144-HEAD ABLATION SCAN")
print("=" * 70)

head_results = {}
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        name = "L%dH%d" % (layer, head)
        hooks = [("blocks.%d.attn.hook_z" % layer,
                  partial(scale_head, head_idx=head, alpha=0.0))]

        result = eval_bias(model, discovery_prompts, hooks=hooks)

        # Quick PPL check (50 sentences for speed)
        ppl_arr = eval_wikitext_ppl(model, hooks=hooks, n_sentences=50)
        ppl_mean = float(np.mean(ppl_arr))
        ppl_change = (ppl_mean - baseline_ppl) / baseline_ppl * 100

        abs_reduction = (baseline["abs_bias"] - result["abs_bias"]) / baseline["abs_bias"] * 100
        signed_delta = result["signed_bias"] - baseline["signed_bias"]

        head_results[name] = {
            "layer": layer,
            "head": head,
            "signed_bias": result["signed_bias"],
            "abs_bias": result["abs_bias"],
            "total_gender_mass": result["total_gender_mass"],
            "stereotype_preference": result["stereotype_preference"],
            "abs_reduction_pct": float(abs_reduction),
            "signed_delta": float(signed_delta),
            "ppl_mean": float(ppl_mean),
            "ppl_change_pct": float(ppl_change),
        }

        if (layer * 12 + head) % 24 == 0:
            print("  Progress: %d/144 heads scanned..." % (layer * 12 + head))

print("  144/144 heads scanned.")

# ═══════════════════════════════════════════════════════════════
# RANK BY SEPARABILITY (high bias reduction, low PPL cost)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TOP 20 HEADS BY ABSOLUTE BIAS REDUCTION")
print("=" * 70)

sorted_by_reduction = sorted(head_results.items(),
                              key=lambda x: x[1]["abs_reduction_pct"],
                              reverse=True)

print("\n%-10s %10s %10s %10s %10s %10s" % (
    "Head", "AbsBiasRed", "SignedΔ", "PPL%Δ", "GenderMass", "StereoPref"))
print("-" * 62)

for name, d in sorted_by_reduction[:20]:
    separable = "✓" if d["ppl_change_pct"] < 5 else "✗"
    print("%-10s %+9.1f%% %+9.4f %+9.1f%% %10.4f %9.1f%% %s" % (
        name, d["abs_reduction_pct"], d["signed_delta"],
        d["ppl_change_pct"], d["total_gender_mass"],
        d["stereotype_preference"] * 100, separable))

# ═══════════════════════════════════════════════════════════════
# TOP 10: SEPARABLE HEADS (bias reduction > 5%, PPL < 5%)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SEPARABLE HEADS (>5% bias reduction, <5% PPL increase)")
print("=" * 70)

separable = [(n, d) for n, d in sorted_by_reduction
             if d["abs_reduction_pct"] > 5 and d["ppl_change_pct"] < 5]

print("Found %d separable heads (out of 144)" % len(separable))
for name, d in separable[:10]:
    print("  %-10s bias red: %+.1f%%, PPL Δ: %+.1f%%, signed Δ: %+.4f" % (
        name, d["abs_reduction_pct"], d["ppl_change_pct"], d["signed_delta"]))

# ═══════════════════════════════════════════════════════════════
# L10H9 POSITION
# ═══════════════════════════════════════════════════════════════
l10h9_rank = [i for i, (n, _) in enumerate(sorted_by_reduction) if n == "L10H9"]
l10h9_rank = l10h9_rank[0] + 1 if l10h9_rank else "NOT FOUND"

print("\n" + "=" * 70)
print("L10H9 REDISCOVERY CHECK")
print("=" * 70)
if "L10H9" in head_results:
    d = head_results["L10H9"]
    print("  L10H9 rank by abs bias reduction: %s / 144" % l10h9_rank)
    print("  Abs bias reduction: %.1f%%" % d["abs_reduction_pct"])
    print("  PPL change:         %+.1f%%" % d["ppl_change_pct"])
    print("  Signed delta:       %+.4f" % d["signed_delta"])

    top1 = sorted_by_reduction[0][0]
    if top1 == "L10H9":
        print("\n  RESULT: L10H9 IS #1 — successfully rediscovered")
    elif l10h9_rank <= 5:
        print("\n  RESULT: L10H9 is #%s (top 5) — rediscovered" % l10h9_rank)
    else:
        print("\n  RESULT: L10H9 is #%s — %s is the new top head" % (l10h9_rank, top1))
        print("  REPORTING HONESTLY: new top head differs from original discovery")

# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP CIs ON TOP 10
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BOOTSTRAP CIs FOR TOP 10 HEADS")
print("=" * 70)

top10_names = [n for n, _ in sorted_by_reduction[:10]]
for name in top10_names:
    layer, head = head_results[name]["layer"], head_results[name]["head"]
    hooks = [("blocks.%d.attn.hook_z" % layer,
              partial(scale_head, head_idx=head, alpha=0.0))]

    result = eval_bias(model, discovery_prompts, hooks=hooks)
    reductions = (baseline["_abs_scores"] - result["_abs_scores"]) / (baseline["_abs_scores"] + 1e-10) * 100
    lo, hi = bootstrap_ci(reductions, n_boot=10000)
    mean_red = head_results[name]["abs_reduction_pct"]
    head_results[name]["abs_reduction_ci"] = [float(lo), float(hi)]
    print("  %-10s: %.1f%% [%.1f%%, %.1f%%]" % (name, mean_red, lo, hi))

# ═══════════════════════════════════════════════════════════════
# FULL EVAL ON TOP 3 SEPARABLE HEADS
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FULL EVALUATION ON TOP 3 SEPARABLE HEADS")
print("=" * 70)

top3_separable = [(n, d) for n, d in separable[:3]]
full_eval_results = {}

for name, d in top3_separable:
    layer, head = d["layer"], d["head"]
    hooks = [("blocks.%d.attn.hook_z" % layer,
              partial(scale_head, head_idx=head, alpha=0.0))]
    print("\n--- %s (full eval on discovery) ---" % name)
    fe = full_eval(model, hooks=hooks, split="discovery", capability="full", verbose=True)
    print_results(fe, label="%s ablated" % name)
    full_eval_results[name] = results_to_json(fe)

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
save_data = {
    "baseline": {
        "signed_bias": baseline["signed_bias"],
        "abs_bias": baseline["abs_bias"],
        "total_gender_mass": baseline["total_gender_mass"],
        "wikitext_ppl": baseline_ppl,
    },
    "head_scan": head_results,
    "top20_by_reduction": [(n, head_results[n]) for n, _ in sorted_by_reduction[:20]],
    "n_separable": len(separable),
    "separable_heads": [(n, head_results[n]) for n, _ in separable],
    "l10h9_rank": l10h9_rank,
    "top3_full_eval": full_eval_results,
}

with open(RESULTS_DIR / "head_rediscovery_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ GPT discovery complete.")
