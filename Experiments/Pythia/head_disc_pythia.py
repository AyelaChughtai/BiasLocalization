"""
(Pythia-2.8B): Head Re-Discovery on Discovery Set

Full head ablation scan on Pythia-2.8B (32 layers × 32 heads = 1024 heads)
using the first 50 discovery-set prompts for speed.

Architecture differences from GPT-2:
  - 32 layers, 32 heads (vs 12 layers, 12 heads)
  - Rotary positional embeddings (RoPE)
  - GPT-NeoX tokenizer (same " he"/" she" tokens as GPT-2 BPE)
  - hook_z works identically — same ablation logic applies
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

device = "cuda" if torch.cuda.is_available() else "cpu"

print("=" * 70)
print("(PYTHIA-2.8B): HEAD SCAN ON DISCOVERY SET")
print("=" * 70)

from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained("pythia-2.8b", device=device)

n_layers = model.cfg.n_layers  # 32
n_heads  = model.cfg.n_heads   # 32
total_heads = n_layers * n_heads

print("Model: pythia-2.8b")
print("  n_layers: %d" % n_layers)
print("  n_heads:  %d" % n_heads)
print("  Total heads to scan: %d" % total_heads)

# ── Only use first 50 prompts for speed ──────────────────────────────
all_discovery_prompts = get_prompts("discovery")
discovery_prompts = all_discovery_prompts[:50]
print("\nDiscovery prompts (capped at 50): %d  (full set: %d)" % (
    len(discovery_prompts), len(all_discovery_prompts)))


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
# FULL HEAD SCAN  (1024 heads — bias + quick PPL per head)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("%d-HEAD ABLATION SCAN (%d layers × %d heads)" % (total_heads, n_layers, n_heads))
print("=" * 70)

head_results = {}
for layer in range(n_layers):
    for head in range(n_heads):
        name = "L%dH%d" % (layer, head)
        hooks = [("blocks.%d.attn.hook_z" % layer,
                  partial(scale_head, head_idx=head, alpha=0.0))]

        result = eval_bias(model, discovery_prompts, hooks=hooks)

        ppl_arr    = eval_wikitext_ppl(model, hooks=hooks, n_sentences=50)
        ppl_mean   = float(np.mean(ppl_arr))
        ppl_change = (ppl_mean - baseline_ppl) / baseline_ppl * 100

        abs_reduction = (baseline["abs_bias"] - result["abs_bias"]) / baseline["abs_bias"] * 100
        signed_delta  = result["signed_bias"] - baseline["signed_bias"]

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

        scanned = layer * n_heads + head + 1
        if scanned % 64 == 0 or scanned == total_heads:
            print("  Progress: %d/%d heads scanned..." % (scanned, total_heads))

print("  %d/%d heads scanned." % (total_heads, total_heads))

# ═══════════════════════════════════════════════════════════════
# RANK BY ABSOLUTE BIAS REDUCTION
# ═══════════════════════════════════════════════════════════════
sorted_by_reduction = sorted(head_results.items(),
                              key=lambda x: x[1]["abs_reduction_pct"],
                              reverse=True)

print("\n" + "=" * 70)
print("TOP 20 HEADS BY ABSOLUTE BIAS REDUCTION")
print("=" * 70)

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
# SEPARABLE HEADS  (bias reduction > 5%, PPL change < 5%)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SEPARABLE HEADS (>5%% bias reduction, <5%% PPL increase)")
print("=" * 70)

separable = [(n, d) for n, d in sorted_by_reduction
             if d["abs_reduction_pct"] > 5 and d["ppl_change_pct"] < 5]

print("Found %d separable heads (out of %d)" % (len(separable), total_heads))
for name, d in separable[:10]:
    print("  %-10s bias red: %+.1f%%, PPL Δ: %+.1f%%, signed Δ: %+.4f" % (
        name, d["abs_reduction_pct"], d["ppl_change_pct"], d["signed_delta"]))

# ═══════════════════════════════════════════════════════════════
# LAYER-LEVEL LOCALIZATION
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("LAYER-LEVEL LOCALIZATION")
print("=" * 70)

print("\nMean and max bias reduction by layer:")
for layer in range(n_layers):
    layer_heads = [r for _, r in head_results.items() if r["layer"] == layer]
    avg_red = float(np.mean([r["abs_reduction_pct"] for r in layer_heads]))
    max_red = float(max(r["abs_reduction_pct"] for r in layer_heads))
    max_h   = next(r["head"] for r in layer_heads if r["abs_reduction_pct"] == max_red)
    bar = "#" * max(0, int(avg_red * 2))
    print("  L%-3d avg=%+6.1f%%  max=%+6.1f%% (H%-2d)  %s" % (
        layer, avg_red, max_red, max_h, bar))

# ═══════════════════════════════════════════════════════════════
# TOP HEAD CHECK  (analogous to L10H9 check for GPT-2)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("TOP HEAD ANALYSIS")
print("=" * 70)

top1_name, top1_data = sorted_by_reduction[0]
print("  #1 head: %s" % top1_name)
print("  Abs bias reduction: %.1f%%" % top1_data["abs_reduction_pct"])
print("  PPL change:         %+.1f%%" % top1_data["ppl_change_pct"])
print("  Signed delta:       %+.4f"   % top1_data["signed_delta"])
print("  Layer depth:        %.0f%% (layer %d / %d)" % (
    top1_data["layer"] / n_layers * 100, top1_data["layer"], n_layers))

# Compare to GPT-2 L10H9 reference
gpt2_depth = 10 / 12 * 100
print("\n  GPT-2 L10H9 reference: layer 10/12 = %.0f%% depth, bias_red=32.9%%, ppl_Δ=+1.1%%" % gpt2_depth)
print("  Pythia top head:       layer %d/%d  = %.0f%% depth, bias_red=%.1f%%, ppl_Δ=%+.1f%%" % (
    top1_data["layer"], n_layers,
    top1_data["layer"] / n_layers * 100,
    top1_data["abs_reduction_pct"],
    top1_data["ppl_change_pct"]))

# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP CIs ON TOP 10
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("BOOTSTRAP CIs FOR TOP 10 HEADS")
print("=" * 70)

top10_names = [n for n, _ in sorted_by_reduction[:10]]
for name in top10_names:
    layer = head_results[name]["layer"]
    head  = head_results[name]["head"]
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

top3 = separable[:3] if len(separable) >= 3 else sorted_by_reduction[:3]
if not separable:
    print("  No separable heads found — running full eval on top-3 by bias reduction instead.")

full_eval_results = {}
for name, d in top3:
    layer = d["layer"]
    head  = d["head"]
    hooks = [("blocks.%d.attn.hook_z" % layer,
              partial(scale_head, head_idx=head, alpha=0.0))]
    print("\n--- %s (full eval on discovery set) ---" % name)
    fe = full_eval(model, hooks=hooks, split="discovery", capability="full", verbose=True)
    print_results(fe, label="%s ablated" % name)
    full_eval_results[name] = results_to_json(fe)

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
save_data = {
    "model": "pythia-2.8b",
    "n_layers": n_layers,
    "n_heads": n_heads,
    "n_prompts": len(discovery_prompts),
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
    "top1_head": top1_name,
    "gpt2_reference": {
        "head": "L10H9",
        "layer_depth_pct": 10 / 12 * 100,
        "bias_reduction_pct": 32.9,
        "ppl_change_pct": 1.1,
    },
    "top3_full_eval": full_eval_results,
}

with open(RESULTS_DIR / "head_rediscovery_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("\nResults saved to %s" % RESULTS_DIR)
print("\n✓ Pythia discovery complete.")