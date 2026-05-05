# Cross-Model SAE Gender Bias Experiments

Replicates the GPT-2 SAE feature-level male skew / occupation bias decoupling experiment (from `exp_sae_male_skew.ipynb`) across multiple model architectures.

## Structure

```
sae_cross_model/
├── model_configs.py      # Model-specific configs (SAE release, hook names, layers)
├── data.py               # Prompts, occupations, templates (shared across models)
├── metrics.py            # All metric functions + SAE hooks (model-agnostic)
├── run_experiment.ipynb  # Main notebook — change MODEL_KEY to switch models
└── README.md
```

## How to Run

1. Open `run_experiment.ipynb`
2. Set `MODEL_KEY` in the first cell:
   - `"gpt2"` — reproduces the original experiment
   - `"pythia-70m"` — Pythia-70m-deduped (the only Pythia with SAE-Lens SAEs)
   - `"olmo-1b"` — **not yet runnable** (no SAEs available)
3. Run all cells

## Model Coverage

| Model | TransformerLens | SAE-Lens SAEs | Status |
|-------|----------------|---------------|--------|
| GPT-2 Small | ✅ | ✅ `gpt2-small-res-jb` | ✅ Runnable |
| Pythia-70m-deduped | ✅ | ✅ `pythia-70m-deduped-res-sm` | ✅ Runnable |
| OLMo-1B | ✅ | ❌ None published | ⏳ Stub only |

### Why Pythia-70m?

SAE-Lens only has published residual stream SAEs for Pythia-70m-deduped (6 layers, 512 d_model). No larger Pythia models have SAEs in the SAE-Lens registry. This is smaller than GPT-2-small (124M), so expect:
- Weaker gender bias signals
- Potentially no decoupled features (which is itself an informative result)
- The experiment tests whether bias-mass entanglement is architecture-specific or general

### What about OLMo?

TransformerLens supports OLMo-1B, but no one has published SAE-Lens-compatible SAEs for OLMo yet. Options:
1. Wait for community SAEs
2. Train your own (SAE-Lens supports this, but it's a project in itself)
3. Use a non-SAE intervention method (e.g., DAS / linear probes)

## Key Differences from the Original Notebook

1. **Config-driven**: All model-specific parameters in `model_configs.py` — no hardcoded layer numbers, hook names, or SAE releases in the experiment code.

2. **Auto-built interventions**: Instead of hardcoding feature IDs (F4077, F837, F23772 from GPT-2), the notebook automatically selects the top decoupled / bias-reducing features found during discovery. This means you don't need to know GPT-2's feature IDs to run on Pythia.

3. **Gender token verification**: Explicitly checks that `he/she` token strings resolve to valid single tokens in each model's tokenizer (they should, since both use BPE, but worth verifying).

4. **All metric functions take token IDs as args**: No global `male_token_ids` / `female_token_ids` — everything is passed explicitly.

## Adding a New Model

Add a new config dict in `model_configs.py` following the existing schema. You need:
- `model_id`: TransformerLens model name
- `sae_release` / `sae_id`: SAE-Lens release + SAE identifier
- `hook_name`: TransformerLens hook point for the intervention
- `sae_layer`: Layer index for activation collection
- `male_token_strings` / `female_token_strings`: Gender tokens for this tokenizer
