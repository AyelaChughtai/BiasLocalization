"""
neuron_sweep_autoresearch.py

Applies the bias_neurons cheap-ablation methodology to the bias-autoresearch
prompt format.  Three stages controlled by --stage:

  proxy  – FAST first pass.  Scores every MLP neuron via Direct Logit
            Attribution: one caching forward pass per prompt to get each
            neuron's mean activation at the last position, then computes
            proxy_score_k = mean_act_k × (W_out_l[k,:] @ gender_direction)
            where gender_direction = Σ W_U[:,male_ids] − Σ W_U[:,female_ids].
            No individual ablation forward passes.  Run as a Slurm array job
            with --neuron-start / --neuron-stop to parallelise across shards.
            Produces a ranking CSV sorted by |proxy_score|.

  scan   – Exact first pass (same as cheap_neuron_ablation_sweep in
            bias_neurons).  Sweeps each MLP neuron individually with a
            mean-zero hook and measures the true bias-metric change.  Slower
            than proxy but does not require the linear approximation.
            Suitable for small models (GPT-2, Pythia-70M); for large models
            (Gemma-2-2B) use the proxy stage instead.

  ablate – Loads a ranking CSV from proxy or scan, takes the top --top-n
            neurons by |proxy_score| or abs_bias_delta, runs exact mean-zero
            and zero ablation, and writes per-prompt + aggregate CSVs with
            absolute_bias, signed_bias, pmale, pfemale, gender_mass
            before/after for each neuron.

  group  – For the top-10 neurons by (a) abs_bias_delta, (b) signed_bias_delta,
            (c) -signed_bias_delta, tries prefix groups of size 1..10 ablated
            simultaneously and records all bias metrics before/after.

Gender evaluation delegates to eval_utils.eval_bias(), which sums
probabilities over he/him/his/himself and she/her/hers/herself — consistent
with every other experiment in this project.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Make eval_utils importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import load_splits, get_gender_ids, eval_bias

# ──────────────────────────────────────────────────────────────────────────────
# PATHS / CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "neuron_sweep"

DEFAULT_SPLITS_FILES = [
    DATA_DIR / "splits.json",
    DATA_DIR / "splits2.json",
    DATA_DIR / "splits3.json",
]

DEFAULT_MODEL = "gpt2"
DEFAULT_SEED = 1729
DEFAULT_BATCH_SIZE = 128


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def neuron_id(layer: int, neuron: int) -> str:
    return f"L{int(layer)}N{int(neuron)}"


def parse_neuron_label(label: str) -> tuple[int, int]:
    v = str(label).strip()
    if not v.startswith("L") or "N" not in v:
        raise ValueError(f"Expected label like L5N1319, got {label!r}")
    layer_s, neuron_s = v[1:].split("N", 1)
    return int(layer_s), int(neuron_s)


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)


def append_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


# ──────────────────────────────────────────────────────────────────────────────
# PROMPT GENERATION
# ──────────────────────────────────────────────────────────────────────────────

def generate_all_prompts(splits_files: list[Path]) -> list[str]:
    """Return deduplicated list of 'The {occ} {tmpl}' prompts from all splits files."""
    seen: set[str] = set()
    prompts: list[str] = []
    for path in splits_files:
        splits = load_splits(str(path))
        for split_data in splits.values():
            for occ in split_data["occupations"]:
                for tmpl in split_data["templates"]:
                    text = f"The {occ} {tmpl}"
                    if text not in seen:
                        seen.add(text)
                        prompts.append(text)
    return prompts


# ──────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name: str, device: str):
    from transformer_lens import HookedTransformer
    print(f"[model] Loading {model_name} on {device}")
    # No default_prepend_bos=False — matches the autoresearch convention so that
    # our tokenisation is consistent with eval_utils.eval_bias().
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    print(f"[model] n_layers={model.cfg.n_layers}, d_mlp={model.cfg.d_mlp}")
    return model


# ──────────────────────────────────────────────────────────────────────────────
# HOOKS
# ──────────────────────────────────────────────────────────────────────────────

def make_neuron_hook(layer: int, neuron: int, value: float) -> tuple[str, object]:
    """Hook that sets the last-position activation of one MLP neuron to value."""
    hook_name = f"blocks.{layer}.mlp.hook_post"
    nidx = int(neuron)
    val = float(value)

    def _hook(activation: torch.Tensor, hook) -> torch.Tensor:
        updated = activation.clone()
        updated[:, -1, nidx] = val
        return updated

    return hook_name, _hook


def make_group_hooks(neurons: list[tuple[int, int]], mean_values: dict[tuple[int, int], float]) -> list[tuple[str, object]]:
    """One hook per layer that simultaneously sets multiple neurons to their means."""
    by_layer: dict[int, dict[int, float]] = defaultdict(dict)
    for layer, neuron in neurons:
        by_layer[layer][neuron] = mean_values[(layer, neuron)]

    hooks = []
    for layer, nmap in by_layer.items():
        hook_name = f"blocks.{layer}.mlp.hook_post"
        nidxs = list(nmap.keys())
        vals = [nmap[n] for n in nidxs]

        def _hook(activation: torch.Tensor, hook, _nidxs=nidxs, _vals=vals) -> torch.Tensor:
            updated = activation.clone()
            for nidx, val in zip(_nidxs, _vals):
                updated[:, -1, nidx] = val
            return updated

        hooks.append((hook_name, _hook))
    return hooks


# ──────────────────────────────────────────────────────────────────────────────
# MEAN ACTIVATION COMPUTATION (not in eval_utils — needed for mean ablation)
# ──────────────────────────────────────────────────────────────────────────────

def compute_neuron_means(
    model,
    prompts: list[str],
    all_neurons: list[tuple[int, int]],
    batch_size: int,
    device: str,
) -> dict[tuple[int, int], float]:
    """Compute mean MLP post-activation at the last token position over all prompts."""
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, neuron in all_neurons:
        by_layer[layer].append(neuron)

    sums: dict[tuple[int, int], float] = {k: 0.0 for k in all_neurons}
    counts: dict[tuple[int, int], int] = {k: 0 for k in all_neurons}

    # Group prompts by token length so we can batch them
    tokenised: list[tuple[torch.Tensor, int]] = []
    for prompt in prompts:
        toks = model.to_tokens(prompt).squeeze(0)  # (seq,) — uses model default BOS setting
        tokenised.append((toks, int(toks.shape[0]) - 1))

    by_length: dict[int, list[tuple[torch.Tensor, int]]] = defaultdict(list)
    for toks, last_pos in tokenised:
        by_length[toks.shape[0]].append((toks, last_pos))

    names_filter = lambda name: name.endswith("mlp.hook_post")

    with torch.inference_mode():
        for length, group in by_length.items():
            for start in range(0, len(group), batch_size):
                batch = group[start: start + batch_size]
                token_batch = torch.stack([t for t, _ in batch]).to(device)
                last_positions = torch.tensor([p for _, p in batch], device=device)
                _, cache = model.run_with_cache(token_batch, names_filter=names_filter, return_type="logits")
                bidx = torch.arange(token_batch.shape[0], device=device)
                for layer, layer_neurons in by_layer.items():
                    hook_name = f"blocks.{layer}.mlp.hook_post"
                    acts = cache[hook_name][bidx, last_positions, :]  # (B, d_mlp)
                    acts_np = acts.detach().cpu().numpy().astype(np.float64)
                    for neuron in layer_neurons:
                        col = acts_np[:, neuron]
                        sums[(layer, neuron)] += float(col.sum())
                        counts[(layer, neuron)] += len(batch)

    return {k: sums[k] / counts[k] for k in all_neurons if counts[k] > 0}


def load_mean_cache(path: Path) -> dict[tuple[int, int], float]:
    if not path.exists():
        return {}
    mc = pd.read_csv(path)
    return {(int(r.layer), int(r.neuron)): float(r.mean_activation) for r in mc.itertuples(index=False)}


def save_mean_cache(path: Path, new_values: dict[tuple[int, int], float]) -> None:
    new_rows = pd.DataFrame([{"layer": l, "neuron": n, "mean_activation": v} for (l, n), v in new_values.items()])
    if path.exists():
        old = pd.read_csv(path)
        combined = pd.concat([old, new_rows], ignore_index=True).drop_duplicates(["layer", "neuron"], keep="last")
    else:
        combined = new_rows
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def metrics_from_eval_bias(result: dict) -> dict[str, float]:
    """Extract the scalar bias metrics from an eval_bias() result dict."""
    return {
        "signed_bias": result["signed_bias"],
        "abs_bias": result["abs_bias"],
        "gender_mass": result["total_gender_mass"],
        "stereotype_preference": result["stereotype_preference"],
    }


def per_prompt_from_eval_bias(result: dict) -> pd.DataFrame:
    """Reconstruct per-prompt pmale / pfemale from the internal score arrays."""
    signed = result["_signed_scores"]    # pm - pf
    mass = result["_mass_scores"]        # pm + pf
    pmale = (mass + signed) / 2.0
    pfemale = (mass - signed) / 2.0
    return pd.DataFrame({
        "pmale": pmale,
        "pfemale": pfemale,
        "signed_bias": signed,
        "abs_bias": result["_abs_scores"],
        "gender_mass": mass,
    })


def aggregate_condition(
    model_name: str,
    dataset_tag: str,
    layer: int,
    neuron: int,
    ablation_mode: str,
    baseline: dict,
    after: dict,
) -> dict:
    row = {
        "model": model_name,
        "dataset": dataset_tag,
        "layer": layer,
        "neuron_id": neuron_id(layer, neuron),
        "neuron": neuron,
        "ablation_mode": ablation_mode,
        "n_prompts": after.get("n_prompts", baseline.get("n_prompts")),
    }
    for metric in ["signed_bias", "abs_bias", "gender_mass", "stereotype_preference"]:
        key_before = "total_gender_mass" if metric == "gender_mass" else metric
        key_after = "total_gender_mass" if metric == "gender_mass" else metric
        row[f"{metric}_before"] = baseline[key_before]
        row[f"{metric}_after"] = after[key_after]
    # pmale / pfemale from signed + mass
    row["pmale_before"] = (baseline["total_gender_mass"] + baseline["signed_bias"]) / 2.0
    row["pfemale_before"] = (baseline["total_gender_mass"] - baseline["signed_bias"]) / 2.0
    row["pmale_after"] = (after["total_gender_mass"] + after["signed_bias"]) / 2.0
    row["pfemale_after"] = (after["total_gender_mass"] - after["signed_bias"]) / 2.0
    row["signed_bias_delta"] = row["signed_bias_before"] - row["signed_bias_after"]
    row["abs_bias_delta"] = row["abs_bias_before"] - row["abs_bias_after"]
    row["gender_mass_delta"] = row["gender_mass_after"] - row["gender_mass_before"]
    row["abs_bias_reduction_pct"] = (
        100.0 * row["abs_bias_delta"] / row["abs_bias_before"]
        if row["abs_bias_before"] else math.nan
    )
    row["male_pref_rate_delta"] = row["stereotype_preference_after"] - row["stereotype_preference_before"]
    return row


# ──────────────────────────────────────────────────────────────────────────────
# PROXY STAGE — DLA fast pass (O(n_prompts) forward passes, no per-neuron loop)
# ──────────────────────────────────────────────────────────────────────────────

def stage_proxy(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    male_ids, female_ids = get_gender_ids(model)

    splits_files = [Path(p) for p in args.splits_files]
    prompts = generate_all_prompts(splits_files)
    print(f"[proxy] {len(prompts)} unique prompts from {len(splits_files)} splits file(s)")

    all_neurons = [
        (layer, neuron)
        for layer in range(model.cfg.n_layers)
        for neuron in range(model.cfg.d_mlp)
    ]
    if args.neuron_start is not None or args.neuron_stop is not None:
        start = args.neuron_start or 0
        stop = args.neuron_stop if args.neuron_stop is not None else len(all_neurons)
        all_neurons = all_neurons[start:stop]
    print(f"[proxy] Neuron range {args.neuron_start or 0}..{args.neuron_stop or 'end'} ({len(all_neurons)} neurons)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: one caching pass per prompt-batch to get mean activations
    print("[proxy] Computing mean activations …")
    t0 = time.perf_counter()
    mean_values = compute_neuron_means(model, prompts, all_neurons, args.batch_size, device)
    print(f"[proxy] Mean activations computed in {time.perf_counter() - t0:.1f}s")

    # Step 2: gender direction in residual stream  (d_model,)
    with torch.no_grad():
        m_ids = torch.tensor(male_ids, dtype=torch.long)
        f_ids = torch.tensor(female_ids, dtype=torch.long)
        # W_U shape: (d_model, d_vocab)
        gender_dir = (model.W_U[:, m_ids].sum(-1) - model.W_U[:, f_ids].sum(-1)).float().cpu()

    # Step 3: proxy_score_k = mean_act_k × (W_out[layer][k, :] @ gender_dir)
    by_layer: dict[int, list[int]] = defaultdict(list)
    for layer, neuron in all_neurons:
        by_layer[layer].append(neuron)

    rows = []
    for layer, layer_neurons in sorted(by_layer.items()):
        with torch.no_grad():
            # W_out shape: (d_mlp, d_model)
            w_out = model.blocks[layer].mlp.W_out.float().cpu()
            logit_weights = (w_out @ gender_dir)  # (d_mlp,)
        for neuron in layer_neurons:
            mean_act = mean_values.get((layer, neuron), float("nan"))
            glw = float(logit_weights[neuron])
            proxy = mean_act * glw
            rows.append({
                "layer": layer,
                "neuron": neuron,
                "neuron_id": neuron_id(layer, neuron),
                "mean_activation": mean_act,
                "gender_logit_weight": glw,
                "proxy_score": proxy,
                "abs_proxy_score": abs(proxy),
            })

    proxy_df = pd.DataFrame.from_records(rows).sort_values("abs_proxy_score", ascending=False)
    aggregate_path = out_dir / "aggregate_scan.csv"
    proxy_df.to_csv(aggregate_path, index=False)
    print(f"[proxy] Wrote {len(proxy_df)} rows → {aggregate_path}")

    save_json(out_dir / "metadata_proxy.json", {
        "stage": "proxy",
        "model": args.model,
        "splits_files": [str(p) for p in splits_files],
        "n_prompts": len(prompts),
        "n_neurons": len(all_neurons),
        "neuron_start": args.neuron_start,
        "neuron_stop": args.neuron_stop,
        "seed": args.seed,
        "output_dir": str(out_dir),
    })
    print(f"[proxy] Done. Output: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# SCAN STAGE — batched for speed (36 864 neurons)
# ──────────────────────────────────────────────────────────────────────────────

def scan_batch(
    model,
    token_batches: list[tuple[torch.Tensor, torch.Tensor]],  # (token_batch, last_positions)
    layer: int,
    neuron: int,
    ablation_value: float,
    male_ids: list[int],
    female_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one ablated forward pass per length-group batch. Returns (pmale, pfemale, mass) arrays."""
    hook_name, hook_fn = make_neuron_hook(layer, neuron, ablation_value)
    pmales, pfemales, masses = [], [], []
    m_ids = torch.tensor(male_ids)
    f_ids = torch.tensor(female_ids)
    with torch.inference_mode():
        for token_batch, _ in token_batches:
            logits = model.run_with_hooks(token_batch, return_type="logits", fwd_hooks=[(hook_name, hook_fn)])
            last_logits = logits[:, -1, :]
            probs = torch.softmax(last_logits.float(), dim=-1)
            m_ids_dev = m_ids.to(probs.device)
            f_ids_dev = f_ids.to(probs.device)
            pm = probs[:, m_ids_dev].sum(-1).detach().cpu().numpy().astype(np.float64)
            pf = probs[:, f_ids_dev].sum(-1).detach().cpu().numpy().astype(np.float64)
            pmales.append(pm)
            pfemales.append(pf)
            masses.append(pm + pf)
    pmale = np.concatenate(pmales)
    pfemale = np.concatenate(pfemales)
    mass = np.concatenate(masses)
    return pmale, pfemale, mass


def stage_scan(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    # Reuse eval_utils for gender token IDs
    male_ids, female_ids = get_gender_ids(model)

    splits_files = [Path(p) for p in args.splits_files]
    prompts = generate_all_prompts(splits_files)
    print(f"[scan] {len(prompts)} unique prompts from {len(splits_files)} splits file(s)")

    # Pre-tokenise and group by length for efficient batching
    tokenised: list[tuple[torch.Tensor, int]] = []
    for prompt in prompts:
        toks = model.to_tokens(prompt).squeeze(0)
        tokenised.append((toks, int(toks.shape[0]) - 1))

    by_length: dict[int, list[torch.Tensor]] = defaultdict(list)
    for toks, _ in tokenised:
        by_length[toks.shape[0]].append(toks)

    # Build batches (token_batch tensor per chunk)
    token_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for length, toks_list in sorted(by_length.items()):
        for start in range(0, len(toks_list), args.batch_size):
            chunk = toks_list[start: start + args.batch_size]
            token_batch = torch.stack(chunk).to(device)
            # last_positions not used in scan_batch (hooks use [:, -1, :]) but kept for shape info
            token_batches.append((token_batch, None))

    # Baseline — use eval_bias from eval_utils for consistency
    print("[scan] Computing baseline with eval_utils.eval_bias …")
    baseline_result = eval_bias(model, prompts)
    print(f"[scan] Baseline signed_bias={baseline_result['signed_bias']:.5f}  "
          f"abs_bias={baseline_result['abs_bias']:.5f}  "
          f"gender_mass={baseline_result['total_gender_mass']:.5f}")

    bl_pmale = (baseline_result["total_gender_mass"] + baseline_result["signed_bias"]) / 2.0
    bl_pfemale = (baseline_result["total_gender_mass"] - baseline_result["signed_bias"]) / 2.0
    # Per-prompt arrays for computing aggregate after ablation
    bl_signed = baseline_result["_signed_scores"]
    bl_mass = baseline_result["_mass_scores"]
    bl_abs = baseline_result["_abs_scores"]

    # Neuron range
    all_neurons = [
        (layer, neuron)
        for layer in range(model.cfg.n_layers)
        for neuron in range(model.cfg.d_mlp)
    ]
    if args.neuron_start is not None or args.neuron_stop is not None:
        start = args.neuron_start or 0
        stop = args.neuron_stop if args.neuron_stop is not None else len(all_neurons)
        all_neurons = all_neurons[start:stop]
    print(f"[scan] Neuron range {args.neuron_start or 0}..{args.neuron_stop or 'end'}  ({len(all_neurons)} neurons)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate_path = out_dir / "aggregate_scan.csv"
    mean_cache_path = out_dir / "neuron_means.csv"

    # Mean activations (for mean ablation mode)
    mean_values = load_mean_cache(mean_cache_path)
    if "mean" in args.ablation_modes:
        missing = [n for n in all_neurons if n not in mean_values]
        if missing:
            print(f"[scan] Computing mean activations for {len(missing)} neurons …")
            computed = compute_neuron_means(model, prompts, missing, args.batch_size, device)
            mean_values.update(computed)
            save_mean_cache(mean_cache_path, computed)
        else:
            print(f"[scan] Mean activations loaded from cache ({len(mean_values)} entries)")

    # Resume support
    completed: set[tuple[int, int, str]] = set()
    if aggregate_path.exists():
        prev = pd.read_csv(aggregate_path, usecols=["layer", "neuron", "ablation_mode"])
        for r in prev.itertuples(index=False):
            completed.add((int(r.layer), int(r.neuron), str(r.ablation_mode)))

    n_total = len(all_neurons) * len(args.ablation_modes)
    idx = 0
    t0 = time.perf_counter()

    for layer, neuron in all_neurons:
        for mode in args.ablation_modes:
            idx += 1
            if (layer, neuron, mode) in completed:
                continue

            abl_val = 0.0 if mode == "zero" else mean_values[(layer, neuron)]

            pmale_after, pfemale_after, mass_after = scan_batch(
                model, token_batches, layer, neuron, abl_val, male_ids, female_ids
            )
            signed_after = pmale_after - pfemale_after
            abs_after = np.abs(signed_after)

            agg = {
                "model": "gpt2",
                "dataset": "autoresearch_all_splits",
                "layer": layer,
                "neuron_id": neuron_id(layer, neuron),
                "neuron": neuron,
                "ablation_mode": mode,
                "n_prompts": len(prompts),
                "pmale_before": bl_pmale,
                "pfemale_before": bl_pfemale,
                "signed_bias_before": float(np.mean(bl_signed)),
                "abs_bias_before": float(np.mean(bl_abs)),
                "gender_mass_before": float(np.mean(bl_mass)),
                "male_pref_rate_before": float(np.mean(bl_signed > 0)),
                "pmale_after": float(np.mean(pmale_after)),
                "pfemale_after": float(np.mean(pfemale_after)),
                "signed_bias_after": float(np.mean(signed_after)),
                "abs_bias_after": float(np.mean(abs_after)),
                "gender_mass_after": float(np.mean(mass_after)),
                "male_pref_rate_after": float(np.mean(signed_after > 0)),
            }
            agg["signed_bias_delta"] = agg["signed_bias_before"] - agg["signed_bias_after"]
            agg["abs_bias_delta"] = agg["abs_bias_before"] - agg["abs_bias_after"]
            agg["gender_mass_delta"] = agg["gender_mass_after"] - agg["gender_mass_before"]
            agg["abs_bias_reduction_pct"] = (
                100.0 * agg["abs_bias_delta"] / agg["abs_bias_before"]
                if agg["abs_bias_before"] else math.nan
            )

            append_csv(aggregate_path, pd.DataFrame.from_records([agg]))
            completed.add((layer, neuron, mode))

            if idx % 500 == 0:
                elapsed = time.perf_counter() - t0
                eta = (n_total - idx) / (idx / elapsed) if idx else float("inf")
                print(f"[scan] {idx}/{n_total}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s  last={neuron_id(layer, neuron)} {mode}")

    # Rankings
    if aggregate_path.exists():
        agg_df = pd.read_csv(aggregate_path)
        agg_df["rank_abs_bias_delta"] = agg_df["abs_bias_delta"].rank(ascending=False, method="min").astype(int)
        agg_df["rank_signed_bias_delta"] = agg_df["signed_bias_delta"].rank(ascending=False, method="min").astype(int)
        agg_df.to_csv(aggregate_path, index=False)
        agg_df.sort_values("abs_bias_delta", ascending=False).to_csv(out_dir / "ranking_abs_bias_delta.csv", index=False)
        agg_df.sort_values("signed_bias_delta", ascending=False).to_csv(out_dir / "ranking_signed_bias_delta.csv", index=False)
        agg_df.sort_values("signed_bias_delta", ascending=True).to_csv(out_dir / "ranking_reverse_signed_bias_delta.csv", index=False)

    save_json(out_dir / "metadata_scan.json", {
        "stage": "scan",
        "model": args.model,
        "splits_files": [str(p) for p in splits_files],
        "n_prompts": len(prompts),
        "n_neurons": len(all_neurons),
        "neuron_start": args.neuron_start,
        "neuron_stop": args.neuron_stop,
        "ablation_modes": args.ablation_modes,
        "baseline_signed_bias": baseline_result["signed_bias"],
        "baseline_abs_bias": baseline_result["abs_bias"],
        "baseline_gender_mass": baseline_result["total_gender_mass"],
        "seed": args.seed,
        "output_dir": str(out_dir),
    })
    print(f"[scan] Done. Output: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# ABLATE STAGE — exact, per-prompt detail via eval_utils.eval_bias()
# ──────────────────────────────────────────────────────────────────────────────

def stage_ablate(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    splits_files = [Path(p) for p in args.splits_files]
    prompts = generate_all_prompts(splits_files)
    print(f"[ablate] {len(prompts)} prompts")

    # Baseline via eval_utils
    print("[ablate] Computing baseline with eval_utils.eval_bias …")
    baseline = eval_bias(model, prompts)
    print(f"[ablate] Baseline: signed={baseline['signed_bias']:.5f}  "
          f"abs={baseline['abs_bias']:.5f}  mass={baseline['total_gender_mass']:.5f}")

    # Load ranking — supports both proxy format (abs_proxy_score) and scan format (abs_bias_delta)
    ranking_path = Path(args.ranking_file)
    ranking = pd.read_csv(ranking_path)
    if "ablation_mode" in ranking.columns:
        ranking = ranking[ranking["ablation_mode"] == "mean"]
    sort_col = "abs_proxy_score" if "abs_proxy_score" in ranking.columns else "abs_bias_delta"
    ranking = ranking.sort_values(sort_col, ascending=False)
    ranked_top_neurons = [
        (rank, int(r.layer), int(r.neuron))
        for rank, r in enumerate(ranking.head(args.top_n).itertuples(index=False), start=1)
    ]
    top_neurons = [(layer, neuron) for _, layer, neuron in ranked_top_neurons]
    rank_start = max(1, args.rank_start or 1)
    rank_end = min(args.rank_end or args.top_n, len(ranked_top_neurons))
    ranked_subset = [
        (rank, layer, neuron)
        for rank, layer, neuron in ranked_top_neurons
        if rank_start <= rank <= rank_end
    ]
    top_neurons_subset = [(layer, neuron) for _, layer, neuron in ranked_subset]
    print(f"[ablate] Top-{args.top_n} neurons: {[neuron_id(l,n) for l,n in top_neurons[:5]]} …")
    print(f"[ablate] Rank window: {rank_start}..{rank_end} ({len(ranked_subset)} neurons)")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_prompt_path = out_dir / "per_prompt_ablation.csv"
    aggregate_path = out_dir / "aggregate_ablation.csv"
    mean_cache_path = out_dir / "neuron_means.csv"

    # Means
    mean_values = load_mean_cache(mean_cache_path)
    missing = [n for n in top_neurons_subset if n not in mean_values]
    if missing:
        print(f"[ablate] Computing mean activations for {len(missing)} neurons …")
        computed = compute_neuron_means(model, prompts, missing, args.batch_size, device)
        mean_values.update(computed)
        save_mean_cache(mean_cache_path, computed)

    # Baseline per-prompt rows
    bl_pp = per_prompt_from_eval_bias(baseline)
    bl_pp.columns = [c + "_before" for c in bl_pp.columns]
    bl_pp.insert(0, "prompt", prompts)

    # Resume
    completed: set[tuple[int, int, str]] = set()
    if aggregate_path.exists():
        prev = pd.read_csv(aggregate_path, usecols=["layer", "neuron", "ablation_mode"])
        for r in prev.itertuples(index=False):
            completed.add((int(r.layer), int(r.neuron), str(r.ablation_mode)))

    for rank, layer, neuron in ranked_subset:
        for mode in args.ablation_modes:
            if (layer, neuron, mode) in completed:
                print(f"[ablate] rank {rank}/{len(top_neurons)} skip {neuron_id(layer, neuron)} {mode}")
                continue

            abl_val = 0.0 if mode == "zero" else mean_values[(layer, neuron)]
            hook_name, hook_fn = make_neuron_hook(layer, neuron, abl_val)
            print(f"[ablate] rank {rank}/{len(top_neurons)} {neuron_id(layer, neuron)} {mode} (value={abl_val:.4f})")

            # eval_bias handles the full loop and multi-token gender sum
            after = eval_bias(model, prompts, hooks=[(hook_name, hook_fn)])

            # Per-prompt table
            after_pp = per_prompt_from_eval_bias(after)
            after_pp.columns = [c + "_after" for c in after_pp.columns]
            pp = pd.concat([bl_pp, after_pp], axis=1)
            pp.insert(0, "ablation_mode", mode)
            pp.insert(0, "neuron", neuron)
            pp.insert(0, "neuron_id", neuron_id(layer, neuron))
            pp.insert(0, "layer", layer)
            append_csv(per_prompt_path, pp)

            agg = aggregate_condition(args.model, "autoresearch_all_splits", layer, neuron, mode, baseline, after)
            agg["rank"] = rank
            append_csv(aggregate_path, pd.DataFrame.from_records([agg]))
            completed.add((layer, neuron, mode))

    # Rankings
    if aggregate_path.exists():
        agg_df = pd.read_csv(aggregate_path)
        agg_df.sort_values("abs_bias_delta", ascending=False).to_csv(out_dir / "ranking_abs_bias_delta.csv", index=False)
        agg_df.sort_values("signed_bias_delta", ascending=False).to_csv(out_dir / "ranking_signed_bias_delta.csv", index=False)
        agg_df.sort_values("signed_bias_delta", ascending=True).to_csv(out_dir / "ranking_reverse_signed_bias_delta.csv", index=False)

    save_json(out_dir / "metadata_ablate.json", {
        "stage": "ablate",
        "model": args.model,
        "splits_files": [str(p) for p in splits_files],
        "n_prompts": len(prompts),
        "top_n": args.top_n,
        "rank_start": rank_start,
        "rank_end": rank_end,
        "ranking_file": str(ranking_path),
        "ablation_modes": args.ablation_modes,
        "baseline_signed_bias": baseline["signed_bias"],
        "baseline_abs_bias": baseline["abs_bias"],
        "baseline_gender_mass": baseline["total_gender_mass"],
        "seed": args.seed,
        "output_dir": str(out_dir),
    })
    print(f"[ablate] Done. Output: {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# GROUP STAGE — eval_utils.eval_bias() with multi-neuron hooks
# ──────────────────────────────────────────────────────────────────────────────

def stage_group(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    splits_files = [Path(p) for p in args.splits_files]
    prompts = generate_all_prompts(splits_files)
    print(f"[group] {len(prompts)} prompts")

    # Baseline
    print("[group] Computing baseline with eval_utils.eval_bias …")
    baseline = eval_bias(model, prompts)

    ranking_path = Path(args.ranking_file)
    ranking = pd.read_csv(ranking_path)
    if "ablation_mode" in ranking.columns:
        ranking = ranking[ranking["ablation_mode"] == "mean"]

    # Support both proxy format (abs_proxy_score / proxy_score) and scan format (abs_bias_delta / signed_bias_delta)
    is_proxy = "abs_proxy_score" in ranking.columns
    abs_col = "abs_proxy_score" if is_proxy else "abs_bias_delta"
    signed_col = "proxy_score" if is_proxy else "signed_bias_delta"

    top10_abs = [(int(r.layer), int(r.neuron)) for r in ranking.sort_values(abs_col, ascending=False).head(10).itertuples(index=False)]
    top10_signed = [(int(r.layer), int(r.neuron)) for r in ranking.sort_values(signed_col, ascending=False).head(10).itertuples(index=False)]
    top10_rev = [(int(r.layer), int(r.neuron)) for r in ranking.sort_values(signed_col, ascending=True).head(10).itertuples(index=False)]

    all_needed = list({n for lst in [top10_abs, top10_signed, top10_rev] for n in lst})
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_cache_path = out_dir / "neuron_means.csv"

    mean_values = load_mean_cache(mean_cache_path)
    missing = [n for n in all_needed if n not in mean_values]
    if missing:
        print(f"[group] Computing means for {len(missing)} neurons …")
        computed = compute_neuron_means(model, prompts, missing, args.batch_size, device)
        mean_values.update(computed)
        save_mean_cache(mean_cache_path, computed)

    results = []
    for criterion_name, ordered_list in [
        ("abs_bias", top10_abs),
        ("signed_bias", top10_signed),
        ("reverse_signed_bias", top10_rev),
    ]:
        print(f"\n[group] === {criterion_name} ===")
        print(f"[group] Ordered: {[neuron_id(l,n) for l,n in ordered_list]}")
        for k in range(1, len(ordered_list) + 1):
            group = ordered_list[:k]
            # eval_utils.eval_bias handles the full multi-token gender evaluation
            hooks = make_group_hooks(group, mean_values)
            after = eval_bias(model, prompts, hooks=hooks)
            agg = {
                "criterion": criterion_name,
                "k": k,
                "neurons": ";".join(neuron_id(l, n) for l, n in group),
                "n_prompts": len(prompts),
                "signed_bias_before": baseline["signed_bias"],
                "abs_bias_before": baseline["abs_bias"],
                "gender_mass_before": baseline["total_gender_mass"],
                "stereotype_preference_before": baseline["stereotype_preference"],
                "pmale_before": (baseline["total_gender_mass"] + baseline["signed_bias"]) / 2.0,
                "pfemale_before": (baseline["total_gender_mass"] - baseline["signed_bias"]) / 2.0,
                "signed_bias_after": after["signed_bias"],
                "abs_bias_after": after["abs_bias"],
                "gender_mass_after": after["total_gender_mass"],
                "stereotype_preference_after": after["stereotype_preference"],
                "pmale_after": (after["total_gender_mass"] + after["signed_bias"]) / 2.0,
                "pfemale_after": (after["total_gender_mass"] - after["signed_bias"]) / 2.0,
            }
            agg["signed_bias_delta"] = agg["signed_bias_before"] - agg["signed_bias_after"]
            agg["abs_bias_delta"] = agg["abs_bias_before"] - agg["abs_bias_after"]
            agg["gender_mass_delta"] = agg["gender_mass_after"] - agg["gender_mass_before"]
            agg["abs_bias_reduction_pct"] = (
                100.0 * agg["abs_bias_delta"] / agg["abs_bias_before"]
                if agg["abs_bias_before"] else math.nan
            )
            results.append(agg)
            print(f"[group]   k={k:2d}  abs_bias_delta={agg['abs_bias_delta']:+.5f}  "
                  f"signed_bias_delta={agg['signed_bias_delta']:+.5f}  "
                  f"gender_mass_delta={agg['gender_mass_delta']:+.5f}")

    pd.DataFrame.from_records(results).to_csv(out_dir / "group_ablation_results.csv", index=False)
    save_json(out_dir / "metadata_group.json", {
        "stage": "group",
        "model": args.model,
        "splits_files": [str(p) for p in splits_files],
        "n_prompts": len(prompts),
        "ranking_file": str(ranking_path),
        "top10_abs": [neuron_id(l, n) for l, n in top10_abs],
        "top10_signed": [neuron_id(l, n) for l, n in top10_signed],
        "top10_rev": [neuron_id(l, n) for l, n in top10_rev],
        "baseline_signed_bias": baseline["signed_bias"],
        "baseline_abs_bias": baseline["abs_bias"],
        "baseline_gender_mass": baseline["total_gender_mass"],
        "seed": args.seed,
        "output_dir": str(out_dir),
    })
    print(f"\n[group] Done. Output: {out_dir}")


def stage_combo(args: argparse.Namespace) -> None:
    """Test all C(10,5)=252 combinations of 5 neurons from the top-10 by abs_bias.

    Finds the 5-neuron subset that maximises abs_bias_delta. Resume-safe: skips
    combinations already written to combo_ablation_results.csv.
    """
    from itertools import combinations as _combinations

    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    splits_files = [Path(p) for p in args.splits_files]
    prompts = generate_all_prompts(splits_files)
    print(f"[combo] {len(prompts)} prompts")

    print("[combo] Computing baseline …")
    baseline = eval_bias(model, prompts)

    ranking_path = Path(args.ranking_file)
    ranking = pd.read_csv(ranking_path)
    if "ablation_mode" in ranking.columns:
        ranking = ranking[ranking["ablation_mode"] == "mean"]

    is_proxy = "abs_proxy_score" in ranking.columns
    abs_col = "abs_proxy_score" if is_proxy else "abs_bias_delta"

    top10 = [
        (int(r.layer), int(r.neuron))
        for r in ranking.sort_values(abs_col, ascending=False).head(10).itertuples(index=False)
    ]
    print(f"[combo] Top-10: {[neuron_id(l, n) for l, n in top10]}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mean_cache_path = out_dir / "neuron_means.csv"

    mean_values = load_mean_cache(mean_cache_path)
    missing = [n for n in top10 if n not in mean_values]
    if missing:
        print(f"[combo] Computing means for {len(missing)} neurons …")
        computed = compute_neuron_means(model, prompts, missing, args.batch_size, device)
        mean_values.update(computed)
        save_mean_cache(mean_cache_path, computed)

    out_csv = out_dir / "combo_ablation_results.csv"
    done_keys: set[str] = set()
    if out_csv.exists():
        try:
            done_keys = set(pd.read_csv(out_csv)["neurons"].tolist())
            print(f"[combo] Resuming — {len(done_keys)} combos already done")
        except Exception:
            pass

    all_combos = list(_combinations(range(10), 5))
    n_total = len(all_combos)
    print(f"[combo] Testing C(10,5)={n_total} combinations …")

    for i, idxs in enumerate(all_combos):
        group = [top10[idx] for idx in idxs]
        key = ";".join(neuron_id(l, n) for l, n in group)
        if key in done_keys:
            print(f"[combo] skip {i+1}/{n_total}  {key}")
            continue
        hooks = make_group_hooks(group, mean_values)
        after = eval_bias(model, prompts, hooks=hooks)
        row = {
            "combo_idx": i,
            "neurons": key,
            "ranks": ";".join(str(idx + 1) for idx in idxs),
            "n_prompts": len(prompts),
            "abs_bias_before": baseline["abs_bias"],
            "signed_bias_before": baseline["signed_bias"],
            "gender_mass_before": baseline["total_gender_mass"],
            "abs_bias_after": after["abs_bias"],
            "signed_bias_after": after["signed_bias"],
            "gender_mass_after": after["total_gender_mass"],
        }
        row["abs_bias_delta"] = row["abs_bias_before"] - row["abs_bias_after"]
        row["signed_bias_delta"] = row["signed_bias_before"] - row["signed_bias_after"]
        row["abs_bias_reduction_pct"] = (
            100.0 * row["abs_bias_delta"] / row["abs_bias_before"]
            if row["abs_bias_before"] else math.nan
        )
        append_csv(out_csv, pd.DataFrame.from_records([row]))
        done_keys.add(key)
        print(
            f"[combo] {i+1}/{n_total}  {key}  "
            f"abs_bias_delta={row['abs_bias_delta']:+.5f}  "
            f"reduction={row['abs_bias_reduction_pct']:+.1f}%"
        )

    all_results = pd.read_csv(out_csv)

    # Pick the best combo for each of the three criteria
    best_abs    = all_results.sort_values("abs_bias_delta",    ascending=False).iloc[0]
    best_signed = all_results.sort_values("signed_bias_delta", ascending=False).iloc[0]
    best_rev    = all_results.sort_values("signed_bias_delta", ascending=True).iloc[0]

    winners = pd.DataFrame([
        {"criterion": "abs_bias",            "neurons": best_abs["neurons"],    "ranks": best_abs["ranks"],
         "abs_bias_delta": best_abs["abs_bias_delta"],       "signed_bias_delta": best_abs["signed_bias_delta"]},
        {"criterion": "signed_bias",         "neurons": best_signed["neurons"], "ranks": best_signed["ranks"],
         "abs_bias_delta": best_signed["abs_bias_delta"],    "signed_bias_delta": best_signed["signed_bias_delta"]},
        {"criterion": "reverse_signed_bias", "neurons": best_rev["neurons"],    "ranks": best_rev["ranks"],
         "abs_bias_delta": best_rev["abs_bias_delta"],       "signed_bias_delta": best_rev["signed_bias_delta"]},
    ])
    winners.to_csv(out_dir / "best_combos.csv", index=False)

    for _, w in winners.iterrows():
        print(f"\n[combo] Best for {w['criterion']}: {w['neurons']}")
        print(f"[combo]   ranks={w['ranks']}  abs_bias_delta={w['abs_bias_delta']:+.5f}"
              f"  signed_bias_delta={w['signed_bias_delta']:+.5f}")

    save_json(out_dir / "metadata_combo.json", {
        "stage": "combo",
        "model": args.model,
        "splits_files": [str(p) for p in splits_files],
        "n_prompts": len(prompts),
        "ranking_file": str(ranking_path),
        "top10": [neuron_id(l, n) for l, n in top10],
        "n_combos": n_total,
        "best_abs_bias":            {"neurons": best_abs["neurons"],    "ranks": best_abs["ranks"],    "abs_bias_delta": float(best_abs["abs_bias_delta"])},
        "best_signed_bias":         {"neurons": best_signed["neurons"], "ranks": best_signed["ranks"], "signed_bias_delta": float(best_signed["signed_bias_delta"])},
        "best_reverse_signed_bias": {"neurons": best_rev["neurons"],    "ranks": best_rev["ranks"],    "signed_bias_delta": float(best_rev["signed_bias_delta"])},
        "baseline_abs_bias": baseline["abs_bias"],
        "baseline_signed_bias": baseline["signed_bias"],
        "seed": args.seed,
        "output_dir": str(out_dir),
    })
    print(f"\n[combo] Done. Results: {out_csv}  Winners: {out_dir / 'best_combos.csv'}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Neuron bias sweep for bias-autoresearch prompt format.")
    p.add_argument("--stage", choices=["proxy", "scan", "ablate", "group", "combo"], required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--splits-files", nargs="+", default=[str(p) for p in DEFAULT_SPLITS_FILES],
                   help="Paths to splits JSON files (default: all three in data/).")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--ranking-file", type=str, default=None,
                   help="Aggregate scan CSV used to select top neurons (required for ablate/group).")
    p.add_argument("--top-n", type=int, default=100,
                   help="Number of top neurons to ablate exactly (ablate stage).")
    p.add_argument("--rank-start", type=int, default=None,
                   help="First top-neuron rank to process in ablate stage (1-indexed, inclusive).")
    p.add_argument("--rank-end", type=int, default=None,
                   help="Last top-neuron rank to process in ablate stage (1-indexed, inclusive).")
    p.add_argument("--ablation-modes", nargs="+", choices=["zero", "mean"], default=["mean"],
                   help="Ablation modes for scan and ablate stages.")
    p.add_argument("--neuron-start", type=int, default=None,
                   help="First neuron index in the flat (layer, neuron) enumeration (scan stage).")
    p.add_argument("--neuron-stop", type=int, default=None,
                   help="Exclusive upper bound for neuron index (scan stage).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.output_dir is None:
        # Sanitise model name for use as a directory component (e.g. "google/gemma-2-2b" → "gemma-2-2b")
        model_slug = args.model.split("/")[-1]
        tag = f"{model_slug}_{args.stage}_{ts()}"
        if args.stage in ("scan", "proxy") and args.neuron_start is not None:
            tag += f"_shard_{args.neuron_start}_{args.neuron_stop}"
        args.output_dir = str(RESULTS_DIR / tag)

    print(f"[main] stage={args.stage}  output={args.output_dir}")

    if args.stage == "proxy":
        stage_proxy(args)
    elif args.stage == "scan":
        stage_scan(args)
    elif args.stage == "ablate":
        if not args.ranking_file:
            raise ValueError("--ranking-file is required for the ablate stage")
        stage_ablate(args)
    elif args.stage == "group":
        if not args.ranking_file:
            raise ValueError("--ranking-file is required for the group stage")
        stage_group(args)
    elif args.stage == "combo":
        if not args.ranking_file:
            raise ValueError("--ranking-file is required for the combo stage")
        stage_combo(args)


if __name__ == "__main__":
    main()
