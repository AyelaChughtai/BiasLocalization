"""
GPT-2: Test-Set Final Evaluation

Held-out test set evaluation for the paper's main results table.
Interventions tested (based on discovery results):
  - Baseline
  - L10H9 ablation (α=0.0)
  - L10H9 attenuation (α=0.1 ... 0.9)
  - Multi-head ablation top-5 (L10H9, L11H3, L9H6, L10H0, L8H11)

Model: GPT-2.
Full capability suite with bootstrap CIs on test split.
"""

import torch
import numpy as np
import json
import sys
from pathlib import Path
from functools import partial
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval_utils import (
    full_eval, print_results, results_to_json
)

RESULTS_DIR = Path(__file__).parent / "Results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


def make_interventions(n_heads_per_layer):
    """
    n_heads_per_layer: number of heads (12 for GPT-2)

    Multi-head top-5 uses heads identified in Exp 21 for GPT-2.
    """
    multi_heads = [(10, 9), (11, 3), (9, 6), (10, 0), (8, 11)]

    multi_hooks = [
        ("blocks.%d.attn.hook_z" % l,
         partial(scale_head, head_idx=h, alpha=0.0))
        for l, h in multi_heads
        if h < n_heads_per_layer   # safety guard for different model sizes
    ]

    return {
        "Baseline": None,
        "L10H9 ablation (α=0.0)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.0))
        ],
        "L10H9 attenuation (α=0.1)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.1))
        ],
        "L10H9 attenuation (α=0.2)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.2))
        ],
        "L10H9 attenuation (α=0.3)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.3))
        ],
        "L10H9 attenuation (α=0.4)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.4))
        ],
        "L10H9 attenuation (α=0.5)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.5))
        ],
        "L10H9 attenuation (α=0.6)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.6))
        ],
        "L10H9 attenuation (α=0.7)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.7))
        ],
        "L10H9 attenuation (α=0.8)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.8))
        ],
        "L10H9 attenuation (α=0.9)": [
            ("blocks.10.attn.hook_z", partial(scale_head, head_idx=9, alpha=0.9))
        ],
        "Multi-head ablation (top-5)": multi_hooks,
    }


# ═══════════════════════════════════════════════════════════════
# GPT-2
# ═══════════════════════════════════════════════════════════════
print("=" * 70)
print("EXPERIMENT: GPT TEST")
print("=" * 70)
print("\nInterventions")
print("  1. Baseline")
print("  2. L10H9 ablation (α=0.0)        — primary head, full ablation")
print("  3. L10H9 attenuation (full)     — partial intervention")
print("  4. Multi-head ablation (top-5)   — L10H9 + L11H3 + L9H6 + L10H0 + L8H11")

print("\n" + "=" * 70)
print("GPT-2 — TEST SPLIT")
print("=" * 70)

model_gpt2 = HookedTransformer.from_pretrained("gpt2", device=device)
gpt2_interventions = make_interventions(n_heads_per_layer=model_gpt2.cfg.n_heads)

gpt2_results = {}
for name, hooks in gpt2_interventions.items():
    print("\n" + "-" * 70)
    print("GPT-2 | %s" % name)
    print("-" * 70)
    results = full_eval(
        model_gpt2, hooks=hooks,
        split="test", capability="full",
        n_boot=10000, verbose=True
    )
    print_results(results, label="GPT-2 — %s" % name)
    gpt2_results[name] = results_to_json(results)

del model_gpt2
torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════
# MAIN RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PAPER MAIN RESULTS TABLE (TEST SET)")
print("=" * 70)

# Flatten into a single ordered dict: GPT-2 rows first, then Pythia
col_order = (
    [("GPT-2", n) for n in gpt2_interventions]
)

METRICS = [
    ("signed_bias",           "Signed bias"),
    ("abs_bias",              "Absolute bias"),
    ("total_gender_mass",     "Gender mass"),
    ("stereotype_preference", "Stereo pref"),
    ("wikitext_ppl",          "WikiText PPL"),
    ("lambada_acc",           "LAMBADA acc"),
    ("winogender_male_pref",  "Winogender M%"),
    ("winobias_type1_gap",    "WinoBias T1 gap"),
    ("winobias_type2_gap",    "WinoBias T2 gap"),
    ("gap_overall",           "GAP overall"),
    ("crows_pairs_score",     "CrowS-Pairs"),
]

def fmt_val(v, ci=None):
    if v is None:
        return "N/A"
    s = ("%.4f" % v) if abs(v) < 10 else ("%.2f" % v)
    if ci:
        s += " [%.3f,%.3f]" % (ci[0], ci[1])
    return s

# Header
col_labels = ["%s/%s" % (mdl, name[:14]) for mdl, name in col_order]
print("\n%-22s" % "Metric", end="")
for lbl in col_labels:
    print("  %-22s" % lbl[:22], end="")
print()
print("-" * (22 + 24 * len(col_order)))

for key, label in METRICS:
    print("%-22s" % label, end="")
    for mdl, name in col_order:
        res = gpt2_results if mdl == "GPT-2" else pythia_results
        r   = res.get(name, {})
        v   = r.get(key)
        ci  = r.get(key + "_ci")
        print("  %-22s" % fmt_val(v, ci)[:22], end="")
    print()

# ═══════════════════════════════════════════════════════════════
# DELTA TABLE (change vs baseline within each model)
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("DELTA VS BASELINE (within each model)")
print("=" * 70)

DELTA_METRICS = [
    ("abs_bias",              "Abs bias Δ",        True),   # True = lower is better
    ("total_gender_mass",     "Gender mass Δ",     None),   # neutral (suppression check)
    ("stereotype_preference", "Stereo pref Δ",     True),
    ("wikitext_ppl",          "WikiText PPL Δ",    True),
    ("winobias_type1_gap",    "WinoBias T1 gap Δ", True),
    ("crows_pairs_score",     "CrowS-Pairs Δ",     True),
]

for mdl, results_dict in [("GPT-2", gpt2_results)]:
    if not results_dict:
        continue
    base = results_dict.get("Baseline", {})
    print("\n  %s:" % mdl)
    non_base = [(n, r) for n, r in results_dict.items() if n != "Baseline"]

    header = "  %-22s" % "Metric"
    for n, _ in non_base:
        header += "  %-20s" % n[:20]
    print(header)
    print("  " + "-" * (22 + 22 * len(non_base)))

    for key, label, lower_better in DELTA_METRICS:
        bv = base.get(key)
        if bv is None:
            continue
        print("  %-22s" % label, end="")
        for n, r in non_base:
            v = r.get(key)
            if v is None:
                print("  %-20s" % "N/A", end="")
            else:
                delta = v - bv
                pct   = delta / (abs(bv) + 1e-10) * 100
                marker = ""
                if lower_better is True:
                    marker = "↓" if delta < -0.001 else ("↑" if delta > 0.001 else "=")
                elif lower_better is False:
                    marker = "↑" if delta > 0.001 else ("↓" if delta < -0.001 else "=")
                print("  %-20s" % ("%+.4f (%+.1f%%) %s" % (delta, pct, marker))[:20], end="")
        print()

# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════
save_data = {
    "gpt2":    gpt2_results,
}

with open(RESULTS_DIR / "head_test_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("Results saved to %s" % RESULTS_DIR)
print("\n✓ GPT test complete.")