"""
Gemma-2B: Test-Set Final Evaluation

Held-out test set evaluation for the paper's main results table.
Interventions tested:
  - Baseline
  - L16H6 ablation (α=0.0)
  - L16H6 attenuation sweep (α=0.1 ... 0.9)
  - Multi-head ablation: top-5 separable heads

Discovery results (gemma-2b, 18 layers × 8 heads, first 50 disc prompts):
    L16H6   +48.3%  +0.9%   ← #1
    L2H3    +19.5%  −0.2%
    L16H1   +19.4%  +2.9%
    L7H5    +18.7%  +2.3%
    L16H3   +18.0%  +1.4%
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
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfFolder

sys.path.insert(0, str(Path(__file__).parent.parent))
from eval_utils import (
    full_eval, print_results, results_to_json, get_gender_ids
)

RESULTS_DIR = Path(__file__).parent / "Results"

device = "cuda" if torch.cuda.is_available() else "cpu"
token = HfFolder.get_token()


def scale_head(z, hook, head_idx, alpha):
    z[:, :, head_idx, :] *= alpha
    return z


def build_gemma_interventions():
    """
    L16H6 is the dominant separable head (+48.3% bias_red, +0.9% PPL).
    Top-5 separable heads sweep used for multi-head ablation.
    """

    # top-5 separable heads from discovery scan
    multi_top5 = [
        ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.0)),
        ("blocks.2.attn.hook_z",  partial(scale_head, head_idx=3, alpha=0.0)),
        ("blocks.16.attn.hook_z", partial(scale_head, head_idx=1, alpha=0.0)),
        ("blocks.7.attn.hook_z",  partial(scale_head, head_idx=5, alpha=0.0)),
        ("blocks.16.attn.hook_z", partial(scale_head, head_idx=3, alpha=0.0)),
    ]

    return {
        # ── baseline ────────────────────────────────────────────────────────
        "Baseline": None,

        # ── L16H6 full ablation ──────────────────────────────────────────────
        "L16H6 ablation (α=0.0)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.0))
        ],

        # ── L16H6 attenuation sweep ──────────────────────────────────────────
        "L16H6 attenuation (α=0.1)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.1))
        ],
        "L16H6 attenuation (α=0.2)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.2))
        ],
        "L16H6 attenuation (α=0.3)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.3))
        ],
        "L16H6 attenuation (α=0.4)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.4))
        ],
        "L16H6 attenuation (α=0.5)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.5))
        ],
        "L16H6 attenuation (α=0.6)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.6))
        ],
        "L16H6 attenuation (α=0.7)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.7))
        ],
        "L16H6 attenuation (α=0.8)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.8))
        ],
        "L16H6 attenuation (α=0.9)": [
            ("blocks.16.attn.hook_z", partial(scale_head, head_idx=6, alpha=0.9))
        ],

        # ── multi-head ablation (top-5 separable) ────────────────────────────
        "Multi-head ablation (top-5)": multi_top5,
    }


# ═══════════════════════════════════════════════════════════════
# GEMMA-2B
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GEMMA-2B — TEST SPLIT")
print("=" * 70)

gemma_results = {}
MODEL_NAME = None
model_gemma = None

# Gated load pattern (same as discovery script)
for candidate in ["google/gemma-2-2b", "google/gemma-2b"]:
    try:
        print("Trying to load %s..." % candidate)

        hf_model = AutoModelForCausalLM.from_pretrained(
            candidate,
            token=token,
            torch_dtype=torch.float32,
        )
        hf_tokenizer = AutoTokenizer.from_pretrained(
            candidate,
            token=token,
        )

        model_gemma = HookedTransformer.from_pretrained(
            candidate,
            hf_model=hf_model,
            tokenizer=hf_tokenizer,
            device=device,
        )
        MODEL_NAME = candidate
        print("Loaded %s: %d layers, %d heads, d_model=%d" % (
            candidate, model_gemma.cfg.n_layers,
            model_gemma.cfg.n_heads, model_gemma.cfg.d_model))
        break
    except Exception as e:
        print("  Failed: %s" % str(e)[:200])
        continue

if MODEL_NAME is None:
    raise RuntimeError(
        "Could not load any Gemma variant.\n"
        "Check:\n"
        "  1. huggingface-cli login has been run and token is saved\n"
        "  2. You have accepted the Gemma license at hf.co/google/gemma-2-2b\n"
        "  3. transformers version supports the chosen Gemma variant\n"
        "  4. torch.cuda.is_available() = %s" % torch.cuda.is_available()
    )

# Sanity check: gender token resolution via the same path eval_utils uses
m_ids, f_ids = get_gender_ids(model_gemma)
m_decoded = [model_gemma.tokenizer.decode([i]) for i in m_ids]
f_decoded = [model_gemma.tokenizer.decode([i]) for i in f_ids]
print("Male tokens:  ", list(zip(m_ids, m_decoded)))
print("Female tokens:", list(zip(f_ids, f_decoded)))
assert all(d.strip().lower() in ["he", "him", "his", "himself"] for d in m_decoded), \
    "Male token resolution failed: %s" % m_decoded
assert all(d.strip().lower() in ["she", "her", "hers", "herself"] for d in f_decoded), \
    "Female token resolution failed: %s" % f_decoded
print("Token verification passed.\n")

try:
    gemma_interventions = build_gemma_interventions()

    for name, hooks in gemma_interventions.items():
        print("\n" + "-" * 70)
        print("Gemma | %s" % name)
        print("-" * 70)
        results = full_eval(
            model_gemma, hooks=hooks,
            split="test", capability="full",
            n_boot=10000, verbose=True
        )
        print_results(results, label="Gemma — %s" % name)
        gemma_results[name] = results_to_json(results)

    del model_gemma
    torch.cuda.empty_cache()

except Exception as e:
    print("Gemma evaluation failed: %s" % e)
    import traceback
    traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# MAIN RESULTS TABLE
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("PAPER MAIN RESULTS TABLE (TEST SET)")
print("=" * 70)

col_order = [("Gemma", n) for n in (gemma_results or {})]

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
        r  = gemma_results.get(name, {})
        v  = r.get(key)
        ci = r.get(key + "_ci")
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

for mdl, results_dict in [("Gemma-2-2B", gemma_results)]:
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
    "gemma":  gemma_results,
}

with open(RESULTS_DIR / "head_test_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

print("Results saved to %s" % RESULTS_DIR)
print("\n✓ Gemma test complete.")