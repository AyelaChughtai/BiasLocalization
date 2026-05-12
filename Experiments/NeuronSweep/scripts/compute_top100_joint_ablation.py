"""Ablate all top-100 neurons simultaneously and compute occupation-level bias metrics.

Replicates the exact same approach as the group ablation in neuron_sweep_autoresearch.py
(text prompts, last-token hooks, eval_utils.eval_bias).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

# Add script dir to path so we can import from neuron_sweep_autoresearch
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from neuron_sweep_autoresearch import (
    load_model,
    generate_all_prompts,
    compute_neuron_means,
    make_group_hooks,
    load_mean_cache,
    save_mean_cache,
)
from eval_utils import eval_bias


def parse_neuron_id(s: str) -> tuple[int, int]:
    s = s.strip()
    layer = int(s[1 : s.index("N")])
    neuron = int(s[s.index("N") + 1 :])
    return layer, neuron


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="TransformerLens model name")
    p.add_argument("--neuron-ids", required=True, help="Comma-separated neuron IDs (e.g. L7N3050,L5N1319)")
    p.add_argument("--splits-files", nargs="+", required=True)
    p.add_argument("--mean-cache", type=Path, default=None, help="neuron_means.csv to reuse precomputed means")
    p.add_argument("--output-file", required=True, type=Path)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    print(f"[top100] Loading model {args.model} on {device}")
    model = load_model(args.model, device)

    splits_files = [Path(s) for s in args.splits_files]
    prompts = generate_all_prompts(splits_files)
    print(f"[top100] {len(prompts)} prompts from {len(splits_files)} splits files")

    neurons = [parse_neuron_id(s) for s in args.neuron_ids.split(",")]
    print(f"[top100] Ablating {len(neurons)} neurons jointly")

    # Load cached means if available
    mean_values = {}
    if args.mean_cache and args.mean_cache.exists():
        mean_values = load_mean_cache(args.mean_cache)
        print(f"[top100] Loaded {len(mean_values)} cached means")

    missing = [n for n in neurons if n not in mean_values]
    if missing:
        print(f"[top100] Computing means for {len(missing)} neurons...")
        computed = compute_neuron_means(model, prompts, missing, args.batch_size, device)
        mean_values.update(computed)
        if args.mean_cache:
            save_mean_cache(args.mean_cache, computed)
            print(f"[top100] Saved means to {args.mean_cache}")

    print("[top100] Computing baseline bias...")
    baseline = eval_bias(model, prompts, batch_size=args.batch_size, device=device)
    sb_before = baseline["signed_bias"]
    ab_before = baseline["abs_bias"]
    gm_before = baseline["total_gender_mass"]
    print(f"[top100] Baseline: signed={sb_before:.5f} abs={ab_before:.5f} mass={gm_before:.5f}")

    hooks = make_group_hooks(neurons, mean_values)
    print(f"[top100] Running ablated eval ({len(hooks)} hook(s) across {len({l for l,_ in neurons})} layer(s))...")
    after = eval_bias(model, prompts, hooks=hooks, batch_size=args.batch_size, device=device)
    sb_after = after["signed_bias"]
    ab_after = after["abs_bias"]
    gm_after = after["total_gender_mass"]
    print(f"[top100] After:    signed={sb_after:.5f} abs={ab_after:.5f} mass={gm_after:.5f}")

    abs_reduction_pct = 100.0 * (ab_before - ab_after) / ab_before if ab_before != 0 else 0.0
    signed_pct = 100.0 * (sb_before - sb_after) / abs(sb_before) if sb_before != 0 else 0.0
    gm_pct = 100.0 * (gm_after - gm_before) / gm_before if gm_before != 0 else 0.0

    result = {
        "model": args.model,
        "condition": "top100_mean",
        "n_neurons": len(neurons),
        "n_prompts": len(prompts),
        "signed_bias_before": sb_before,
        "signed_bias_after": sb_after,
        "abs_bias_before": ab_before,
        "abs_bias_after": ab_after,
        "gender_mass_before": gm_before,
        "gender_mass_after": gm_after,
        "abs_bias_reduction_pct": abs_reduction_pct,
        "signed_bias_pct": signed_pct,
        "gender_mass_pct": gm_pct,
    }

    print(f"\n[top100] Results:")
    print(f"  abs_reduction_pct = {abs_reduction_pct:.4f}%")
    print(f"  signed_bias_pct   = {signed_pct:.4f}%")
    print(f"  gender_mass_pct   = {gm_pct:.4f}%")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([result]).to_csv(args.output_file, index=False)
    print(f"[top100] Saved to {args.output_file}")


if __name__ == "__main__":
    main()
