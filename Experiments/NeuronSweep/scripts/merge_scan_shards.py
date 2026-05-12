"""
merge_scan_shards.py

After all scan-, proxy-, or ablate-stage Slurm array tasks finish, merge the
per-shard aggregate CSV files into a single ranked table and write top-N
summary CSVs.

Supports two formats automatically detected from the CSV columns:
  proxy  — columns: layer, neuron, neuron_id, mean_activation,
                    gender_logit_weight, proxy_score, abs_proxy_score
  scan/ablate — columns: layer, neuron, neuron_id, ablation_mode,
                         abs_bias_delta, signed_bias_delta, …

Usage:
  python scripts/merge_scan_shards.py \
      --scan-dir results/neuron_sweep/scan_<ARRAY_JOB_ID> \
      [--out-dir results/neuron_sweep]   # default: same as scan-dir parent
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--scan-dir", required=True, type=Path,
                   help="Directory that contains shard_0/, shard_1/, … subdirs (or CSVs directly).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Where to write merged outputs (default: --scan-dir parent).")
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--ablation-mode", default="mean",
                   help="Mode to use for ranking in scan format (mean or zero).")
    return p.parse_args()


def _is_proxy_format(df: pd.DataFrame) -> bool:
    return "abs_proxy_score" in df.columns and "ablation_mode" not in df.columns


def merge_proxy(merged: pd.DataFrame, out_dir: Path, top_n: int) -> None:
    merged = merged.drop_duplicates(subset=["layer", "neuron"], keep="last")
    print(f"[merge] Total neurons: {len(merged)}")

    merged = merged.sort_values("abs_proxy_score", ascending=False)
    merged["rank_abs_proxy"] = range(1, len(merged) + 1)
    merged["rank_proxy_score"] = merged["proxy_score"].rank(ascending=False, method="min").astype(int)
    merged["rank_reverse_proxy_score"] = merged["proxy_score"].rank(ascending=True, method="min").astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(out_dir / "aggregate_scan_merged.csv", index=False)
    merged.sort_values("abs_proxy_score", ascending=False).to_csv(
        out_dir / "ranking_abs_proxy_score_merged.csv", index=False)
    merged.sort_values("proxy_score", ascending=False).to_csv(
        out_dir / "ranking_proxy_score_merged.csv", index=False)
    merged.sort_values("proxy_score", ascending=True).to_csv(
        out_dir / "ranking_reverse_proxy_score_merged.csv", index=False)

    top = merged.head(top_n)
    top.to_csv(out_dir / f"top{top_n}_abs_proxy_score.csv", index=False)

    print(f"\n[merge] Top-{top_n} by abs_proxy_score:")
    display_cols = ["neuron_id", "layer", "neuron", "mean_activation",
                    "gender_logit_weight", "proxy_score", "abs_proxy_score"]
    display_cols = [c for c in display_cols if c in top.columns]
    print(top[display_cols].to_string(index=False))


def merge_scan(merged: pd.DataFrame, out_dir: Path, top_n: int, ablation_mode: str, source_name: str) -> None:
    merged = merged.drop_duplicates(subset=["layer", "neuron", "ablation_mode"], keep="last")
    print(f"[merge] Total rows: {len(merged)}  unique (layer,neuron,mode): "
          f"{merged[['layer','neuron','ablation_mode']].drop_duplicates().shape[0]}")

    mode_df = merged[merged["ablation_mode"] == ablation_mode].copy()
    print(f"[merge] Rows for mode={ablation_mode}: {len(mode_df)}")

    mode_df["rank_abs_bias_delta"] = mode_df["abs_bias_delta"].rank(ascending=False, method="min").astype(int)
    mode_df["rank_signed_bias_delta"] = mode_df["signed_bias_delta"].rank(ascending=False, method="min").astype(int)
    mode_df["rank_reverse_signed_bias_delta"] = mode_df["signed_bias_delta"].rank(ascending=True, method="min").astype(int)

    out_dir.mkdir(parents=True, exist_ok=True)

    aggregate_name = "aggregate_ablation.csv" if source_name == "aggregate_ablation.csv" else "aggregate_scan_merged.csv"
    ranking_suffix = "" if source_name == "aggregate_ablation.csv" else "_merged"

    merged.to_csv(out_dir / aggregate_name, index=False)
    mode_df.sort_values("abs_bias_delta", ascending=False).to_csv(
        out_dir / f"ranking_abs_bias_delta{ranking_suffix}.csv", index=False)
    mode_df.sort_values("signed_bias_delta", ascending=False).to_csv(
        out_dir / f"ranking_signed_bias_delta{ranking_suffix}.csv", index=False)
    mode_df.sort_values("signed_bias_delta", ascending=True).to_csv(
        out_dir / f"ranking_reverse_signed_bias_delta{ranking_suffix}.csv", index=False)

    top = mode_df.sort_values("abs_bias_delta", ascending=False).head(top_n)
    top.to_csv(out_dir / f"top{top_n}_abs_bias_delta.csv", index=False)

    print(f"\n[merge] Top-{top_n} by abs_bias_delta (mode={ablation_mode}):")
    display_cols = ["neuron_id", "layer", "neuron", "abs_bias_before", "abs_bias_after",
                    "abs_bias_delta", "abs_bias_reduction_pct",
                    "signed_bias_before", "signed_bias_after", "signed_bias_delta",
                    "gender_mass_before", "gender_mass_after"]
    display_cols = [c for c in display_cols if c in top.columns]
    print(top[display_cols].to_string(index=False))


def merge_optional_csvs(scan_dir: Path, out_dir: Path) -> None:
    """Merge auxiliary ablate-stage CSVs when shards provide them."""
    mean_csvs = sorted(scan_dir.glob("shard_*/neuron_means.csv"))
    if mean_csvs:
        means = pd.concat([pd.read_csv(path) for path in mean_csvs], ignore_index=True)
        means = means.drop_duplicates(subset=["layer", "neuron"], keep="last")
        means.sort_values(["layer", "neuron"]).to_csv(out_dir / "neuron_means.csv", index=False)
        print(f"[merge] Merged {len(mean_csvs)} mean-cache CSV(s) → {out_dir / 'neuron_means.csv'}")

    per_prompt_csvs = sorted(scan_dir.glob("shard_*/per_prompt_ablation.csv"))
    if per_prompt_csvs:
        frames = [pd.read_csv(path) for path in per_prompt_csvs]
        pd.concat(frames, ignore_index=True).to_csv(out_dir / "per_prompt_ablation.csv", index=False)
        print(f"[merge] Merged {len(per_prompt_csvs)} per-prompt CSV(s) → {out_dir / 'per_prompt_ablation.csv'}")


def main() -> None:
    args = parse_args()
    scan_dir: Path = args.scan_dir
    out_dir: Path = args.out_dir or scan_dir.parent

    source_name = "aggregate_scan.csv"
    csvs = sorted(scan_dir.glob(source_name))
    if not csvs:
        csvs = sorted(scan_dir.glob(f"shard_*/{source_name}"))
    if not csvs:
        source_name = "aggregate_ablation.csv"
        csvs = sorted(scan_dir.glob(source_name))
    if not csvs:
        csvs = sorted(scan_dir.glob(f"shard_*/{source_name}"))
    if not csvs:
        raise FileNotFoundError(f"No aggregate_scan.csv or aggregate_ablation.csv found under {scan_dir}")

    print(f"[merge] Found {len(csvs)} shard CSV(s)")
    frames = [pd.read_csv(path) for path in csvs]
    merged = pd.concat(frames, ignore_index=True)

    if _is_proxy_format(merged):
        print("[merge] Detected proxy format (DLA scores)")
        merge_proxy(merged, out_dir, args.top_n)
    else:
        print("[merge] Detected scan format (ablation deltas)")
        merge_scan(merged, out_dir, args.top_n, args.ablation_mode, source_name)
        if source_name == "aggregate_ablation.csv":
            merge_optional_csvs(scan_dir, out_dir)

    print(f"\n[merge] Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
