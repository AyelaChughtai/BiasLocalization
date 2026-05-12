# Neuron Sweep Experiments

MLP neuron-level bias localization for GPT-2, Gemma-2-2B, and Pythia-2.8B.
All scripts are run from this directory (`Experiments/NeuronSweep/`).

## Dependencies

```
pip install torch transformer_lens numpy pandas datasets
```

| Package | Role |
|---|---|
| `torch` | Model inference and hooks |
| `transformer_lens` | `HookedTransformer` for activation intervention |
| `numpy` / `pandas` | Numerical computation and CSV I/O |
| `datasets` | WikiText-103 and LAMBADA for capability eval |

## Precomputed results

Precomputed results for all three models are in `results/{model}/`:

```
results/
  gpt2/
  gemma2_2b/
  pythia2_8b/
```

Each model directory contains:

| File | Description |
|---|---|
| `ranking_abs_proxy_score_merged.csv` | All neurons ranked by \|proxy score\| |
| `ranking_proxy_score_merged.csv` | Ranked by signed proxy score |
| `ranking_reverse_proxy_score_merged.csv` | Ranked by reverse signed proxy score |
| `top100_abs_proxy_score.csv` | Top-100 neurons by \|proxy score\| |
| `aggregate_scan.csv` | Raw per-shard proxy scan output |
| `aggregate_scan_merged.csv` | Merged proxy scan (deduplicated) |
| `eval_full_benchmark_results.csv` | Full eval on discovery split (bias + capability) |
| `test_benchmark_results.csv` | Eval on held-out test split |
| `combo_ablation_results.csv` | All C(10,5)=252 five-neuron combinations from top-10 |

## Reproducing the pipeline

All scripts assume the working directory is `Experiments/NeuronSweep/`.
New runs write output to `results/neuron_sweep/` (created automatically).

### Step 1 — Proxy scan (identify candidate neurons)

The proxy stage scores all MLP neurons via Direct Logit Attribution in a single
caching forward pass per prompt — no individual ablations needed.

**GPT-2** (small enough for a single run):

```bash
python scripts/neuron_sweep_autoresearch.py \
    --stage proxy \
    --model gpt2 \
    --splits-files data/splits.json \
    --seed 1729
```

**Gemma-2-2B** (parallelise across neuron shards on a cluster):

```bash
# Each shard covers 2048 neurons (total: 114 688 = 28 layers × 4096 neurons/layer)
# Adjust --neuron-start / --neuron-stop per array task
python scripts/neuron_sweep_autoresearch.py \
    --stage proxy \
    --model google/gemma-2-2b \
    --splits-files data/splits.json \
    --neuron-start 0 --neuron-stop 2048 \
    --seed 1729
```

**Pythia-2.8B**:

```bash
python scripts/neuron_sweep_autoresearch.py \
    --stage proxy \
    --model EleutherAI/pythia-2.8b \
    --splits-files data/splits.json \
    --neuron-start 0 --neuron-stop 2048 \
    --seed 1729
```

### Step 2 — Merge proxy shards

After all shards finish, merge them into a single ranked CSV:

```bash
python scripts/merge_scan_shards.py \
    --scan-dir results/neuron_sweep/<model>_proxy_<timestamp> \
    --top-n 100
```

This writes `ranking_abs_proxy_score_merged.csv` and `top100_abs_proxy_score.csv`
to the same directory.

### Step 3 — Exact ablation of top-100 neurons

```bash
python scripts/neuron_sweep_autoresearch.py \
    --stage ablate \
    --model <model_name> \
    --splits-files data/splits.json \
    --ranking-file results/neuron_sweep/<proxy_dir>/ranking_abs_proxy_score_merged.csv \
    --top-n 100 \
    --ablation-modes mean \
    --output-dir results/neuron_sweep/<model>_ablate \
    --seed 1729
```

### Step 4 — Group ablation (prefix groups of 1–10 from top-10)

```bash
python scripts/neuron_sweep_autoresearch.py \
    --stage group \
    --model <model_name> \
    --splits-files data/splits.json \
    --ranking-file results/neuron_sweep/<ablate_dir>/ranking_abs_bias_delta.csv \
    --output-dir results/neuron_sweep/<model>_group \
    --seed 1729
```

### Step 5 — Combo ablation (best 5-neuron combination from top-10)

Tests all C(10,5)=252 five-neuron subsets and selects the best by each criterion:

```bash
python scripts/neuron_sweep_autoresearch.py \
    --stage combo \
    --model <model_name> \
    --splits-files data/splits.json \
    --ranking-file results/neuron_sweep/<ablate_dir>/ranking_abs_bias_delta.csv \
    --output-dir results/neuron_sweep/<model>_combo \
    --seed 1729
```

### Step 6 — Full eval benchmark (train / discovery split)

Runs `eval_utils.full_eval()` — bias metrics + WikiText-103 PPL + LAMBADA + BLiMP +
WinoGender + WinoBias + CrowS-Pairs — over all ablation conditions:

```bash
python scripts/benchmark_eval_utils_neuron_sweep.py \
    --model <model_name> \
    --ranking-file results/neuron_sweep/<ablate_dir>/ranking_abs_bias_delta.csv \
    --means-file results/neuron_sweep/<ablate_dir>/neuron_means.csv \
    --group-csv results/neuron_sweep/<group_dir>/group_ablation_results.csv \
    --combo-csv results/neuron_sweep/<combo_dir>/combo_ablation_results.csv \
    --top-n 100 \
    --include-individual \
    --split dev \
    --capability full \
    --output-dir results/neuron_sweep/<model>_eval_full \
    --seed 1729
```

### Step 7 — Test-set eval

Same as Step 6 with `--split test`:

```bash
python scripts/benchmark_eval_utils_neuron_sweep.py \
    --model <model_name> \
    --ranking-file results/neuron_sweep/<ablate_dir>/ranking_abs_bias_delta.csv \
    --means-file results/neuron_sweep/<ablate_dir>/neuron_means.csv \
    --group-csv results/neuron_sweep/<group_dir>/group_ablation_results.csv \
    --combo-csv results/neuron_sweep/<combo_dir>/combo_ablation_results.csv \
    --top-n 100 \
    --include-individual \
    --split test \
    --capability full \
    --output-dir results/neuron_sweep/<model>_eval_test \
    --seed 1729
```

## Model identifiers

| Model | TransformerLens name |
|---|---|
| GPT-2 | `gpt2` |
| Gemma-2-2B | `google/gemma-2-2b` |
| Pythia-2.8B | `EleutherAI/pythia-2.8b` |

## Data splits

`data/splits.json` defines the `discovery`, `dev`, and `test` occupation/template
splits used across all stages. The discovery split is used in the sweep and combo
stages; `dev` and `test` are used in the benchmark stages.
