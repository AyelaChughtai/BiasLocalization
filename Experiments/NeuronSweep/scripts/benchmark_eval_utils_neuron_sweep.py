"""
Run neuron-sweep benchmark conditions through eval_utils.full_eval().

This intentionally uses eval_utils' benchmark/data definitions rather than
benchmark_bias_capability.py.  It keeps the neuron-sweep condition construction:
baseline, top-N mean ablation, group rows, combo winners, and individual top-N
mean ablations.  Work can be sharded by 1-indexed condition range.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_utils import full_eval, print_results, results_to_json


def parse_neuron_label(label: str) -> tuple[int, int]:
    value = str(label).strip()
    if not value.startswith("L") or "N" not in value:
        raise ValueError(f"Expected neuron id like L12N345, got {label!r}")
    layer_s, neuron_s = value[1:].split("N", 1)
    return int(layer_s), int(neuron_s)


def neuron_id(layer: int, neuron: int) -> str:
    return f"L{int(layer)}N{int(neuron)}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_name: str, device: str):
    from transformer_lens import HookedTransformer

    print(f"[model] Loading {model_name} on {device}")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    model.eval()
    print(f"[model] n_layers={model.cfg.n_layers}, d_mlp={model.cfg.d_mlp}")
    return model


def make_all_pos_hooks(
    neurons: list[tuple[int, int]],
    mean_values: dict[tuple[int, int], float],
) -> list[tuple[str, object]]:
    """Set selected MLP post-activation neurons to their mean at every position."""
    by_layer: dict[int, dict[int, float]] = defaultdict(dict)
    for layer, neuron in neurons:
        key = (int(layer), int(neuron))
        if key in mean_values:
            by_layer[key[0]][key[1]] = float(mean_values[key])

    hooks: list[tuple[str, object]] = []
    for layer, nmap in sorted(by_layer.items()):
        hook_name = f"blocks.{layer}.mlp.hook_post"
        nidxs = list(nmap.keys())
        vals = [nmap[n] for n in nidxs]

        def _hook(act: torch.Tensor, hook, _nidxs=nidxs, _vals=vals) -> torch.Tensor:
            out = act.clone()
            for nidx, val in zip(_nidxs, _vals):
                out[:, :, nidx] = val
            return out

        hooks.append((hook_name, _hook))
    return hooks


def load_top_neurons(ranking_file: Path, top_n: int) -> list[tuple[int, int]]:
    ranking = pd.read_csv(ranking_file)
    if "ablation_mode" in ranking.columns:
        ranking = ranking[ranking["ablation_mode"] == "mean"]
    sort_col = "abs_proxy_score" if "abs_proxy_score" in ranking.columns else "abs_bias_delta"
    ranking = ranking.sort_values(sort_col, ascending=False)
    return [(int(r.layer), int(r.neuron)) for r in ranking.head(top_n).itertuples(index=False)]


def load_means(means_file: Path) -> dict[tuple[int, int], float]:
    means = pd.read_csv(means_file)
    return {
        (int(r.layer), int(r.neuron)): float(r.mean_activation)
        for r in means.itertuples(index=False)
    }


def build_conditions(args: argparse.Namespace) -> list[dict[str, Any]]:
    top_neurons = load_top_neurons(Path(args.ranking_file), args.top_n)

    conditions: list[dict[str, Any]] = [
        {"condition": "baseline", "condition_type": "baseline", "neurons": []},
        {
            "condition": f"top{args.top_n}_mean",
            "condition_type": "topn",
            "neurons": top_neurons,
        },
    ]

    group_csv = Path(args.group_csv) if args.group_csv else None
    if group_csv and group_csv.exists():
        group_df = pd.read_csv(group_csv)
        for row in group_df.itertuples(index=False):
            label = f"group_{getattr(row, 'criterion')}_k{int(getattr(row, 'k'))}"
            neurons = [parse_neuron_label(x) for x in str(getattr(row, "neurons")).split(";") if x]
            conditions.append({"condition": label, "condition_type": "group", "neurons": neurons})

    combo_csv = Path(args.combo_csv) if args.combo_csv else None
    if combo_csv and combo_csv.exists():
        combo_df = pd.read_csv(combo_csv)
        for row in combo_df.itertuples(index=False):
            label = f"combo_{getattr(row, 'criterion')}"
            neurons = [parse_neuron_label(x) for x in str(getattr(row, "neurons")).split(";") if x]
            conditions.append({"condition": label, "condition_type": "combo", "neurons": neurons})

    if args.include_individual:
        for idx, (layer, neuron) in enumerate(top_neurons, start=1):
            nid = neuron_id(layer, neuron)
            conditions.append({
                "condition": f"neuron_{nid}_mean",
                "condition_type": "individual",
                "neurons": [(layer, neuron)],
                "rank": idx,
                "layer": layer,
                "neuron": neuron,
                "neuron_id": nid,
            })

    return conditions


def json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, new_prefix))
    elif isinstance(value, list):
        out[prefix] = json.dumps(value, default=json_default)
    else:
        out[prefix] = value
    return out


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            with open(path, "a") as fh:
                fh.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_records(rows).to_csv(path, index=False)


def completed_conditions(shard_jsonl: Path) -> set[str]:
    done: set[str] = set()
    if not shard_jsonl.exists():
        return done
    with open(shard_jsonl) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("status") == "ok" and row.get("condition"):
                done.add(str(row["condition"]))
    return done


def merge_shards(out_dir: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted((out_dir / "shards").glob("shard_*.jsonl")):
        with open(path) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No shard rows found under {out_dir / 'shards'}")
    write_csv(out_dir / "eval_utils_benchmark_results.csv", [flatten(r) for r in rows])
    with open(out_dir / "eval_utils_benchmark_results.jsonl", "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=json_default, sort_keys=True) + "\n")
    print(f"[merge] wrote {len(rows)} rows to {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ranking-file", required=True)
    parser.add_argument("--means-file", required=True)
    parser.add_argument("--group-csv")
    parser.add_argument("--combo-csv")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--include-individual", action="store_true")
    parser.add_argument("--condition-start", type=int, default=1, help="1-indexed inclusive")
    parser.add_argument("--condition-end", type=int, default=None, help="1-indexed inclusive")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument(
        "--capability",
        default="full",
        help="Passed to eval_utils.full_eval. Use full, light, false, none, or bias-only.",
    )
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def parse_capability(value: str):
    lowered = str(value).strip().lower()
    if lowered in {"false", "none", "no", "0", "bias", "bias-only", "bias_only"}:
        return False
    if lowered in {"true", "yes", "1"}:
        return "full"
    return value


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    if args.merge_only:
        merge_shards(out_dir)
        return

    set_seed(args.seed)
    capability = parse_capability(args.capability)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    means = load_means(Path(args.means_file))
    all_conditions = build_conditions(args)
    cond_end = min(args.condition_end or len(all_conditions), len(all_conditions))
    cond_start = max(1, args.condition_start)
    selected = all_conditions[cond_start - 1:cond_end]
    if not selected:
        print(f"[bench] no conditions in requested range {cond_start}-{cond_end}")
        return

    shard_name = f"shard_{cond_start:03d}_{cond_end:03d}"
    shard_dir = out_dir / "shards"
    shard_jsonl = shard_dir / f"{shard_name}.jsonl"
    shard_csv = shard_dir / f"{shard_name}.csv"
    shared_jsonl = out_dir / "eval_utils_benchmark_results.live.jsonl"
    meta = {
        "model": args.model,
        "ranking_file": str(Path(args.ranking_file).resolve()),
        "means_file": str(Path(args.means_file).resolve()),
        "group_csv": str(Path(args.group_csv).resolve()) if args.group_csv else None,
        "combo_csv": str(Path(args.combo_csv).resolve()) if args.combo_csv else None,
        "top_n": args.top_n,
        "split": args.split,
        "capability": args.capability,
        "parsed_capability": capability,
        "n_boot": args.n_boot,
        "condition_start": cond_start,
        "condition_end": cond_end,
        "n_total_conditions": len(all_conditions),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)
    with open(shard_dir / f"{shard_name}.metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    done = completed_conditions(shard_jsonl)
    model = load_model(args.model, device)

    rows: list[dict[str, Any]] = []
    for absolute_idx, condition in enumerate(all_conditions[cond_start - 1:cond_end], start=cond_start):
        label = str(condition["condition"])
        if label in done:
            print(f"[bench] skip {label} (already done in {shard_jsonl})")
            continue

        neurons = [(int(l), int(n)) for l, n in condition["neurons"]]
        available = [n for n in neurons if n in means]
        hooks = None if not available else make_all_pos_hooks(available, means)
        if neurons and len(available) < len(neurons):
            missing = len(neurons) - len(available)
            print(f"[bench] warning: {label} missing {missing} mean activations")

        print(f"\n[bench] condition {absolute_idx}/{len(all_conditions)}: {label}")
        base_row = {
            "status": "ok",
            "condition_index": absolute_idx,
            "condition": label,
            "condition_type": condition["condition_type"],
            "model": args.model,
            "split": args.split,
            "capability": args.capability,
            "parsed_capability": capability,
            "n_boot": args.n_boot,
            "neurons": [neuron_id(l, n) for l, n in neurons],
        }
        for key in ("rank", "layer", "neuron", "neuron_id"):
            if key in condition:
                base_row[key] = condition[key]

        try:
            result = full_eval(
                model,
                hooks=hooks,
                split=args.split,
                capability=capability,
                n_boot=args.n_boot,
                verbose=True,
            )
            print_results(result, label=label)
            base_row["results"] = results_to_json(result)
        except Exception as exc:
            base_row["status"] = "error"
            base_row["error"] = repr(exc)
            print(f"[bench] ERROR {label}: {exc!r}")

        rows.append(base_row)
        append_jsonl(shard_jsonl, base_row)
        append_jsonl(shared_jsonl, base_row)
        write_csv(shard_csv, [flatten(r) for r in rows])

    print(f"[bench] wrote shard CSV: {shard_csv}")


if __name__ == "__main__":
    main()
