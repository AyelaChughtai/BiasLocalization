"""
Model configurations for cross-model SAE feature-level gender bias experiments.
Each config specifies everything model-specific so the pipeline stays generic.

Selection rationale:
- GPT-2 small: original model from Exp SAE Male Skew; 12 layers, 768 d_model
- Pythia-70m-deduped: only Pythia with SAE-Lens residual SAEs; 6 layers, 512 d_model
  Smaller than GPT-2, but the *only* Pythia with published SAEs in SAE-Lens.
  Null / negative results here still contribute: they test whether the bias-mass
  entanglement finding generalises to a different architecture + training corpus.
- OLMo: no SAE-Lens SAEs available as of May 2025. Included as a config stub
  in case community SAEs appear, but flagged as not currently runnable.

To add a new model: define a new dict following the schema below and append to MODELS.
"""

# ──────────────────────────────────────────────────────────────────────
# GPT-2 Small  (original experiment)
# ──────────────────────────────────────────────────────────────────────
GPT2_SMALL = {
    "name": "gpt2",
    "display_name": "GPT-2 Small",
    "model_id": "gpt2",                          # TransformerLens model name
    "n_layers": 12,
    "d_model": 768,

    # SAE config
    "sae_release": "gpt2-small-res-jb",
    "sae_id": "blocks.11.hook_resid_pre",         # SAE trained on L11 resid_pre ≈ L10 resid_post
    "hook_name": "blocks.10.hook_resid_post",      # hook point for interventions
    "sae_layer": 10,                               # layer for activation collection

    # Gender tokens (GPT-2 tokenizer)
    "male_token_strings": ["he", " he", "He", " He"],
    "female_token_strings": ["she", " she", "She", " She"],

    # Known features from prior experiments (GPT-2 specific)
    "known_features": {
        "F23440_female": 23440,
        "F16291_male": 16291,
    },
}

# ──────────────────────────────────────────────────────────────────────
# Pythia-70m-deduped
# ──────────────────────────────────────────────────────────────────────
# Architecture: 6 layers, 512 d_model, 8 heads
# SAE: pythia-70m-deduped-res-sm has residual stream SAEs at every layer.
# We target the last layer (block 5) residual post, analogous to using
# the final layers in GPT-2.  The SAE has hook_name blocks.5.hook_resid_post.
#
# IMPORTANT: Pythia-70m is much smaller than GPT-2-small (70M vs 124M).
# Gender bias structure may be weaker / less separable. This is expected
# and still informative — the question is whether the bias-mass entanglement
# pattern replicates, not whether the absolute bias magnitudes match.

PYTHIA_70M = {
    "name": "pythia-70m-deduped",
    "display_name": "Pythia-70M (deduped)",
    "model_id": "EleutherAI/pythia-70m-deduped",
    "n_layers": 6,
    "d_model": 512,

    # SAE config — use the last layer for maximum semantic richness
    "sae_release": "pythia-70m-deduped-res-sm",
    "sae_id": "blocks.5.hook_resid_post",
    "hook_name": "blocks.5.hook_resid_post",
    "sae_layer": 5,

    # Gender tokens — Pythia uses GPT-NeoX tokenizer (same BPE as GPT-2 for these tokens)
    "male_token_strings": ["he", " he", "He", " He"],
    "female_token_strings": ["she", " she", "She", " She"],

    # No known features — discovery phase will find them
    "known_features": {},
}

# ──────────────────────────────────────────────────────────────────────
# Gemma-2-2B
# ──────────────────────────────────────────────────────────────────────
# Architecture: 26 layers, 2304 d_model, 8 heads (4 KV heads, GQA)
# SAE: gemma-2-2b-res-matryoshka-dc has residual stream SAEs at ALL 25 layers.
# These are batch-TopK matryoshka SAEs (chanind/gemma-2-2b-batch-topk-matryoshka-saes)
# with 32k features — well-validated by the community.
#
# We target layer 20 (out of 0-24) for the intervention — this is the ~80% depth
# point, analogous to L10/L12 in GPT-2 (83%) and L5/L6 in Pythia (83%).
# Later layers carry more semantic/gender-relevant features.
#
# Gemma-2-2B at 2.6B params is ~20x larger than GPT-2-small and uses a different
# architecture (GQA, RMSNorm, GeGLU). If the bias-mass entanglement pattern holds
# here too, that's strong evidence for generality.
#
# NOTE: Gemma uses SentencePiece tokenizer. "he"/" he"/"He"/" He" and
# "she"/" she"/"She"/" She" are all single tokens — verified against
# the Gemma tokenizer. The leading-space variants use the SentencePiece
# "▁" prefix internally but model.to_single_token(" he") handles this.
# If token verification fails at runtime, check whether the tokenizer
# needs "▁he" instead of " he".

GEMMA2_2B = {
    "name": "gemma-2-2b",
    "display_name": "Gemma-2 2B",
    "model_id": "google/gemma-2-2b",
    "n_layers": 26,
    "d_model": 2304,

    # SAE config — matryoshka SAEs with full layer coverage
    "sae_release": "gemma-2-2b-res-matryoshka-dc",
    "sae_id": "blocks.20.hook_resid_post",
    "hook_name": "blocks.20.hook_resid_post",
    "sae_layer": 20,

    # Gender tokens — SentencePiece tokenizer
    # If " he" fails token verification, try "▁he" (SentencePiece prefix)
    "male_token_strings": ["he", " he", "He", " He"],
    "female_token_strings": ["she", " she", "She", " She"],

    "known_features": {},
}

# ──────────────────────────────────────────────────────────────────────
# OLMo-1B  (STUB — no SAE-Lens SAEs available)
# ──────────────────────────────────────────────────────────────────────
# TransformerLens supports allenai/OLMo-1B-hf, but there are no published
# SAE-Lens SAEs for OLMo as of May 2025.
#
# Options if you want to include OLMo:
# 1. Train your own SAE (SAE-Lens supports this) — but this is a project in itself.
# 2. Use a different intervention method (e.g., activation patching without SAE
#    decomposition — ablate directions in residual stream directly).
# 3. Wait for community SAEs to appear on HuggingFace.
#
# For now, this config is included so the pipeline is ready if SAEs become available.

OLMO_1B = {
    "name": "olmo-1b",
    "display_name": "OLMo-1B",
    "model_id": "allenai/OLMo-1B-hf",
    "n_layers": 16,
    "d_model": 2048,

    # SAE config — PLACEHOLDER, will need updating
    "sae_release": None,    # <-- No SAE available
    "sae_id": None,
    "hook_name": "blocks.14.hook_resid_post",   # guess: penultimate layer
    "sae_layer": 14,

    "male_token_strings": ["he", " he", "He", " He"],
    "female_token_strings": ["she", " she", "She", " She"],

    "known_features": {},
}

# ──────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────
MODELS = {
    "gpt2": GPT2_SMALL,
    "pythia-70m": PYTHIA_70M,
    "gemma-2-2b": GEMMA2_2B,
    "olmo-1b": OLMO_1B,
}

# Models that are actually runnable (have SAEs)
RUNNABLE_MODELS = {k: v for k, v in MODELS.items() if v["sae_release"] is not None}
