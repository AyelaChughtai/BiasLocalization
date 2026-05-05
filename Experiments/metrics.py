"""
Core metric functions and SAE intervention hooks for cross-model experiments.
All functions take model/sae/token-ids as arguments — no model-specific hardcoding.
"""

import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# ══════════════════════════════════════════════════════════════════════
# Gender probability extraction
# ══════════════════════════════════════════════════════════════════════

def get_gender_probs(logits_at_position, male_token_ids, female_token_ids):
    """Extract male/female probabilities from logits at the prediction position."""
    probs = torch.softmax(logits_at_position, dim=-1)
    p_male = sum(probs[tid].item() for tid in male_token_ids)
    p_female = sum(probs[tid].item() for tid in female_token_ids)
    return p_male, p_female


def _run_model(model, tokens, hook_fn=None, hook_name=None):
    """Run model forward pass, optionally with a hook."""
    if hook_fn is not None and hook_name is not None:
        return model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])
    return model(tokens)


# ══════════════════════════════════════════════════════════════════════
# Bias metrics
# ══════════════════════════════════════════════════════════════════════

def compute_bias_metrics(prompts, model, male_token_ids, female_token_ids,
                         hook_fn=None, hook_name=None):
    """
    Run prompts through model (optionally with a hook) and compute:
      - abs_bias: mean |P(male) - P(female)|
      - signed_bias: mean (P(male) - P(female))
      - gender_mass: mean (P(male) + P(female))
      - per_prompt: list of (p_male, p_female)
    """
    all_p_male, all_p_female = [], []

    for prompt in prompts:
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = _run_model(model, tokens, hook_fn, hook_name)
        p_male, p_female = get_gender_probs(logits[0, -1], male_token_ids, female_token_ids)
        all_p_male.append(p_male)
        all_p_female.append(p_female)

    pm = np.array(all_p_male)
    pf = np.array(all_p_female)

    return {
        "abs_bias": np.mean(np.abs(pm - pf)),
        "signed_bias": np.mean(pm - pf),
        "gender_mass": np.mean(pm + pf),
        "per_prompt": list(zip(all_p_male, all_p_female)),
    }


def compute_bias_metrics_with_ci(prompts, model, male_token_ids, female_token_ids,
                                  hook_fn=None, hook_name=None, n_bootstrap=10000):
    """Compute bias metrics with bootstrap 95% confidence intervals."""
    all_p_male, all_p_female = [], []

    for prompt in tqdm(prompts, desc="Running prompts", leave=False):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = _run_model(model, tokens, hook_fn, hook_name)
        p_male, p_female = get_gender_probs(logits[0, -1], male_token_ids, female_token_ids)
        all_p_male.append(p_male)
        all_p_female.append(p_female)

    pm = np.array(all_p_male)
    pf = np.array(all_p_female)
    n = len(pm)

    abs_bias = np.mean(np.abs(pm - pf))
    signed_bias = np.mean(pm - pf)
    gender_mass = np.mean(pm + pf)

    rng = np.random.default_rng(42)
    boot_abs, boot_signed, boot_mass = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bm, bf = pm[idx], pf[idx]
        boot_abs.append(np.mean(np.abs(bm - bf)))
        boot_signed.append(np.mean(bm - bf))
        boot_mass.append(np.mean(bm + bf))

    return {
        "abs_bias": abs_bias,
        "abs_bias_ci": (np.percentile(boot_abs, 2.5), np.percentile(boot_abs, 97.5)),
        "signed_bias": signed_bias,
        "signed_bias_ci": (np.percentile(boot_signed, 2.5), np.percentile(boot_signed, 97.5)),
        "gender_mass": gender_mass,
        "gender_mass_ci": (np.percentile(boot_mass, 2.5), np.percentile(boot_mass, 97.5)),
        "per_prompt": list(zip(all_p_male, all_p_female)),
    }


def compute_stereotype_amplification(prompts, occupations, templates, model,
                                      male_token_ids, female_token_ids,
                                      hook_fn=None, hook_name=None):
    """
    Stereotype amplification: how much does occupation identity modulate gender preference?
    Returns std of per-occupation signed bias.
    """
    results_by_occ = defaultdict(list)

    for i, prompt in enumerate(prompts):
        occ_idx = i // len(templates)
        occ = occupations[occ_idx]

        tokens = model.to_tokens(prompt, prepend_bos=True)
        with torch.no_grad():
            logits = _run_model(model, tokens, hook_fn, hook_name)
        p_male, p_female = get_gender_probs(logits[0, -1], male_token_ids, female_token_ids)
        results_by_occ[occ].append((p_male, p_female))

    occ_signed = {}
    for occ, pairs in results_by_occ.items():
        pm = np.mean([p[0] for p in pairs])
        pf = np.mean([p[1] for p in pairs])
        occ_signed[occ] = pm - pf

    values = list(occ_signed.values())
    return {
        "amplification": np.std(values),
        "occ_signed": occ_signed,
        "mean_signed": np.mean(values),
    }


# ══════════════════════════════════════════════════════════════════════
# SAE intervention hooks
# ══════════════════════════════════════════════════════════════════════

def make_sae_feature_clamp_hook(sae, feature_id, scale=0.0):
    """
    Hook that clamps a single SAE feature in the residual stream.
    Uses the delta method: only applies the change from the feature modification,
    preserving information not captured by the SAE.
    """
    def hook_fn(activation, hook):
        resid_last = activation[:, -1, :]

        sae_activations = sae.encode(resid_last)
        sae_activations_orig = sae_activations.clone()

        sae_activations[:, feature_id] = sae_activations_orig[:, feature_id] * scale

        reconstructed = sae.decode(sae_activations)
        reconstructed_orig = sae.decode(sae_activations_orig)
        delta = reconstructed - reconstructed_orig

        activation[:, -1, :] = resid_last + delta
        return activation

    return hook_fn


def make_multi_feature_clamp_hook(sae, feature_scales):
    """
    Hook that clamps multiple SAE features simultaneously.
    feature_scales: dict of {feature_id: scale_factor}
    """
    def hook_fn(activation, hook):
        resid_last = activation[:, -1, :]

        sae_activations = sae.encode(resid_last)
        sae_activations_orig = sae_activations.clone()

        for fid, scale in feature_scales.items():
            sae_activations[:, fid] = sae_activations_orig[:, fid] * scale

        reconstructed = sae.decode(sae_activations)
        reconstructed_orig = sae.decode(sae_activations_orig)
        delta = reconstructed - reconstructed_orig

        activation[:, -1, :] = resid_last + delta
        return activation

    return hook_fn


# ══════════════════════════════════════════════════════════════════════
# SAE feature discovery
# ══════════════════════════════════════════════════════════════════════

def get_sae_activations_for_prompts(prompts, model, sae, layer):
    """
    For each prompt, extract SAE feature activations at the last token position
    from the specified layer's residual stream.
    """
    hook_name = f"blocks.{layer}.hook_resid_post"
    all_activations = []

    for prompt in tqdm(prompts, desc="Collecting SAE activations"):
        tokens = model.to_tokens(prompt, prepend_bos=True)
        _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
        resid = cache[hook_name][0, -1]  # [d_model]
        sae_acts = sae.encode(resid.unsqueeze(0))  # [1, d_sae]
        all_activations.append(sae_acts[0].detach().cpu())

    return torch.stack(all_activations)  # [n_prompts, d_sae]


# ══════════════════════════════════════════════════════════════════════
# Capability evaluations
# ══════════════════════════════════════════════════════════════════════

def compute_ppl(sentences, model, hook_fn=None, hook_name=None, max_length=256):
    """Compute perplexity over a set of sentences."""
    total_loss = 0.0
    total_tokens = 0
    for sentence in tqdm(sentences, desc="PPL", leave=False):
        tokens = model.to_tokens(sentence, prepend_bos=True)
        if tokens.shape[1] > max_length:
            tokens = tokens[:, :max_length]
        if tokens.shape[1] < 3:
            continue
        with torch.no_grad():
            logits = _run_model(model, tokens, hook_fn, hook_name)
        shift_logits = logits[:, :-1, :]
        shift_targets = tokens[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_targets.reshape(-1),
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += shift_targets.numel()
    return np.exp(total_loss / total_tokens)


def compute_lambada_acc(examples, model, hook_fn=None, hook_name=None):
    """LAMBADA last-word prediction accuracy."""
    correct, total = 0, 0
    for ex in tqdm(examples, desc="LAMBADA", leave=False):
        text = ex["text"]
        words = text.strip().split()
        if len(words) < 2:
            continue
        last_word = words[-1]
        context = " ".join(words[:-1])

        tokens = model.to_tokens(context, prepend_bos=True)
        with torch.no_grad():
            logits = _run_model(model, tokens, hook_fn, hook_name)
        pred_token = logits[0, -1].argmax().item()
        pred_str = model.to_string([pred_token]).strip()
        if pred_str.lower() == last_word.lower():
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def compute_blimp_acc(examples, model, hook_fn=None, hook_name=None):
    """BLiMP accuracy: fraction where grammatical sentence has higher log-prob."""
    correct, total = 0, 0
    for ex in tqdm(examples, desc="BLiMP", leave=False):
        good = ex.get("sentence_good", ex.get("good", ""))
        bad = ex.get("sentence_bad", ex.get("bad", ""))
        if not good or not bad:
            continue

        scores = []
        for sent in [good, bad]:
            tokens = model.to_tokens(sent, prepend_bos=True)
            with torch.no_grad():
                logits = _run_model(model, tokens, hook_fn, hook_name)
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            target_tokens = tokens[0, 1:]
            score = sum(log_probs[i, target_tokens[i]].item() for i in range(len(target_tokens)))
            scores.append(score)

        if scores[0] > scores[1]:
            correct += 1
        total += 1

    return correct / total if total > 0 else 0.0


def compute_winogender_male_pct(occupations, templates, model,
                                 male_token_ids, female_token_ids,
                                 hook_fn=None, hook_name=None):
    """Compute % of predictions that are male across Winogender occupations."""
    male_count, total = 0, 0
    per_occ = {}

    for occ in occupations:
        occ_male, occ_total = 0, 0
        for template in templates:
            prompt = template.format(occ)
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                logits = _run_model(model, tokens, hook_fn, hook_name)
            p_male, p_female = get_gender_probs(logits[0, -1], male_token_ids, female_token_ids)
            if p_male > p_female:
                male_count += 1
                occ_male += 1
            total += 1
            occ_total += 1
        per_occ[occ] = occ_male / occ_total if occ_total > 0 else 0

    return male_count / total if total > 0 else 0, per_occ


def compute_crows_pairs_score(pairs, model, hook_fn=None, hook_name=None):
    """
    CrowS-Pairs stereotype preference score.
    50% = unbiased (equal preference for stereotypical vs anti-stereotypical).
    """
    stereo_preferred, total = 0, 0

    for pair in tqdm(pairs, desc="CrowS-Pairs", leave=False):
        sent_more = pair.get("sent_more", pair.get("sentence_more", ""))
        sent_less = pair.get("sent_less", pair.get("sentence_less", ""))
        if not sent_more or not sent_less:
            continue

        scores = []
        for sent in [sent_more, sent_less]:
            tokens = model.to_tokens(sent, prepend_bos=True)
            with torch.no_grad():
                logits = _run_model(model, tokens, hook_fn, hook_name)
            log_probs = torch.nn.functional.log_softmax(logits[0, :-1], dim=-1)
            target_tokens = tokens[0, 1:]
            score = sum(log_probs[i, target_tokens[i]].item() for i in range(len(target_tokens)))
            scores.append(score)

        if scores[0] > scores[1]:
            stereo_preferred += 1
        total += 1

    return stereo_preferred / total if total > 0 else 0.0
