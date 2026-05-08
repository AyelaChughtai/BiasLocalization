"""
eval_utils.py — Reusable evaluation module for mech_interp_bias project.

Provides full_eval() returning:
  - Bias metrics (4): signed_bias, abs_bias, total_gender_mass, stereotype_preference
  - General capability (3): wikitext_ppl, lambada_acc, blimp_acc
  - Coreference (2): winogender (preference), winobias (pro/anti accuracy + gap)
  - Bias benchmark (1): crows_pairs_score

All with bootstrap 95% CIs.
"""

import torch
import numpy as np
import json
from pathlib import Path
from functools import lru_cache

# ═══════════════════════════════════════════════════════════════
# DATA SPLITS
# ═══════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def load_splits():
    """Load occupation/template splits from data/splits.json."""
    with open("data/splits.json") as f:
        return json.load(f)


def get_prompts(split_name):
    """Generate all prompts for a given split (discovery/dev/test)."""
    splits = load_splits()
    s = splits[split_name]
    prompts = []
    for occ in s["occupations"]:
        for tmpl in s["templates"]:
            prompts.append("The %s %s" % (occ, tmpl))
    return prompts


# ═══════════════════════════════════════════════════════════════
# TOKEN IDS (cached per model)
# ═══════════════════════════════════════════════════════════════

_gender_ids_cache = {}

def _get_token_id(model, word):
    toks = model.to_tokens(word, prepend_bos=False).squeeze()
    return toks.item() if toks.dim() == 0 else toks[0].item()


def get_gender_ids(model):
    model_name = getattr(model.cfg, "model_name", "default")
    if model_name not in _gender_ids_cache:
        male_words = [" he", " him", " his", " himself"]
        female_words = [" she", " her", " hers", " herself"]
        m_ids = [_get_token_id(model, w) for w in male_words]
        f_ids = [_get_token_id(model, w) for w in female_words]
        _gender_ids_cache[model_name] = (m_ids, f_ids)
    return _gender_ids_cache[model_name]


# ═══════════════════════════════════════════════════════════════
# BIAS METRICS
# ═══════════════════════════════════════════════════════════════

def eval_bias(model, prompts, hooks=None):
    """
    Returns per-prompt scores and aggregate metrics:
    - signed_bias: mean(P(male) - P(female)), positive = male-skewed
    - abs_bias: mean(|P(male) - P(female)|)
    - total_gender_mass: mean(P(male) + P(female))
    - stereotype_preference: fraction of prompts where P(male) > P(female)
    """
    male_ids, female_ids = get_gender_ids(model)
    signed = []
    absolute = []
    mass = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)
        pm = sum(probs[i].item() for i in male_ids)
        pf = sum(probs[i].item() for i in female_ids)
        signed.append(pm - pf)
        absolute.append(abs(pm - pf))
        mass.append(pm + pf)

    signed = np.array(signed)
    absolute = np.array(absolute)
    mass = np.array(mass)

    return {
        "signed_bias": float(np.mean(signed)),
        "abs_bias": float(np.mean(absolute)),
        "total_gender_mass": float(np.mean(mass)),
        "stereotype_preference": float(np.mean(signed > 0)),
        "_signed_scores": signed,
        "_abs_scores": absolute,
        "_mass_scores": mass,
    }


# ═══════════════════════════════════════════════════════════════
# PSEUDO-LOG-LIKELIHOOD
# ═══════════════════════════════════════════════════════════════

def pseudo_log_likelihood(model, text, hooks=None):
    """Sum of log P(token_i | token_0...token_{i-1}) for causal LM."""
    tokens = model.to_tokens(text)
    if tokens.shape[1] <= 1:
        return 0.0
    with torch.no_grad():
        logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
    log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
    target_ids = tokens[0, 1:]
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze()
    if token_log_probs.dim() == 0:
        return token_log_probs.item()
    return token_log_probs.sum().item()


# ═══════════════════════════════════════════════════════════════
# WIKITEXT-103 PERPLEXITY
# ═══════════════════════════════════════════════════════════════

_wikitext_cache = None

def _load_wikitext(n=1000):
    global _wikitext_cache
    if _wikitext_cache is None:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
        # Non-empty, reasonable length
        sents = [s.strip() for s in ds["text"]
                 if len(s.strip()) > 50 and len(s.strip()) < 500
                 and not s.strip().startswith("=")]
        _wikitext_cache = sents
    return _wikitext_cache[:n]


def eval_wikitext_ppl(model, hooks=None, n_sentences=1000):
    """WikiText-103 validation perplexity. Returns array of per-sentence PPLs."""
    sentences = _load_wikitext(n_sentences)
    ppls = []
    for sent in sentences:
        tokens = model.to_tokens(sent)
        if tokens.shape[1] < 3:
            continue
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        log_probs = torch.log_softmax(logits[0, :-1, :], dim=-1)
        targets = tokens[0, 1:]
        loss = -log_probs.gather(1, targets.unsqueeze(1)).squeeze().mean().item()
        ppls.append(np.exp(loss))
    return np.array(ppls)


# ═══════════════════════════════════════════════════════════════
# LAMBADA
# ═══════════════════════════════════════════════════════════════

_lambada_cache = None

def _load_lambada(n=500):
    global _lambada_cache
    if _lambada_cache is None:
        from datasets import load_dataset
        ds = load_dataset("lambada", split="validation")
        _lambada_cache = list(ds)
    return _lambada_cache[:n]


def eval_lambada(model, hooks=None, n_examples=500):
    """LAMBADA last-word prediction accuracy. Returns (accuracy, per_example_results)."""
    examples = _load_lambada(n_examples)
    results = []
    for ex in examples:
        text = ex["text"].strip()
        words = text.split()
        if len(words) < 3:
            continue
        last_word = words[-1]
        prefix = " ".join(words[:-1])

        tokens = model.to_tokens(prefix)
        target_tok = model.to_tokens(" " + last_word, prepend_bos=False).squeeze()
        if target_tok.dim() > 0:
            target_tok = target_tok[0]

        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)

        pred = logits[0, -1, :].argmax().item()
        results.append(1 if pred == target_tok.item() else 0)

    return np.array(results)


# ═══════════════════════════════════════════════════════════════
# BLiMP
# ═══════════════════════════════════════════════════════════════

_blimp_cache = {}

def _load_blimp(config):
    if config not in _blimp_cache:
        from datasets import load_dataset
        _blimp_cache[config] = list(load_dataset("nyu-mll/blimp", config, split="train"))
    return _blimp_cache[config]


def eval_blimp(model, hooks=None):
    """BLiMP minimal pairs. Returns dict of config → accuracy."""
    configs = [
        "anaphor_gender_agreement",
        "regular_plural_subject_verb_agreement_1",
    ]
    results = {}
    per_example = {}
    for config in configs:
        try:
            data = _load_blimp(config)
        except Exception:
            continue
        correct_list = []
        for ex in data:
            pll_good = pseudo_log_likelihood(model, ex["sentence_good"], hooks)
            pll_bad = pseudo_log_likelihood(model, ex["sentence_bad"], hooks)
            correct_list.append(1 if pll_good > pll_bad else 0)
        arr = np.array(correct_list)
        results[config] = float(np.mean(arr))
        per_example[config] = arr

    results["mean"] = float(np.mean([results[c] for c in results]))
    return results, per_example


# ═══════════════════════════════════════════════════════════════
# WINOGENDER — occupation-level gender preference at pronoun position
# ═══════════════════════════════════════════════════════════════

_winogender_cache = None

def _load_winogender():
    global _winogender_cache
    if _winogender_cache is None:
        import pandas as pd
        url = "https://raw.githubusercontent.com/rudinger/winogender-schemas/master/data/all_sentences.tsv"
        df = pd.read_csv(url, sep="\t")
        # Group into (occupation, participant, template_num) → {male: ..., female: ...}
        pairs = {}
        for _, row in df.iterrows():
            parts = row["sentid"].replace(".txt", "").split(".")
            if len(parts) < 4:
                continue
            key = (parts[0], parts[1], parts[2])
            gender = parts[3]
            if gender in ("male", "female"):
                pairs.setdefault(key, {})[gender] = row["sentence"]
        # Keep only complete pairs
        _winogender_cache = {k: v for k, v in pairs.items() if "male" in v and "female" in v}
    return _winogender_cache


def eval_winogender(model, hooks=None):
    """
    Winogender: for each (occupation, participant, template) pair, find the
    pronoun position (first word that differs between male/female variants),
    then compare P(he-variant) vs P(she-variant) at that position.

    Returns:
    - male_pref_rate: fraction of pairs where model prefers male pronoun
    - per_pronoun: accuracy split by pronoun form (he/she, him/her, his/her)
    - per_example: array of 1 (prefers male) / 0 (prefers female)
    """
    pairs = _load_winogender()
    he_id = _get_token_id(model, " he")
    she_id = _get_token_id(model, " she")
    him_id = _get_token_id(model, " him")
    her_id = _get_token_id(model, " her")
    his_id = _get_token_id(model, " his")

    pronoun_map = {
        "he": (he_id, she_id, "he/she"),
        "she": (she_id, he_id, "he/she"),
        "him": (him_id, her_id, "him/her"),
        "her": (her_id, him_id, "him/her"),
        "his": (his_id, her_id, "his/her"),
    }

    results_by_form = {}  # form → list of (prefers_male,)
    all_results = []

    for key, genders in pairs.items():
        male_words = genders["male"].split()
        female_words = genders["female"].split()

        # Find first differing word
        pron_idx = None
        for i in range(min(len(male_words), len(female_words))):
            if male_words[i].lower().rstrip(".,!?;:") != female_words[i].lower().rstrip(".,!?;:"):
                pron_idx = i
                break
        if pron_idx is None:
            continue

        male_pron = male_words[pron_idx].lower().rstrip(".,!?;:")
        if male_pron not in pronoun_map:
            continue

        male_tok_id, female_tok_id, form = pronoun_map[male_pron]

        # Build prefix (same for both variants)
        prefix = " ".join(male_words[:pron_idx])
        if not prefix:
            continue

        tokens = model.to_tokens(prefix)
        with torch.no_grad():
            logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
        probs = torch.softmax(logits[0, -1, :], dim=0)

        p_male = probs[male_tok_id].item()
        p_female = probs[female_tok_id].item()
        prefers_male = int(p_male > p_female)

        all_results.append(prefers_male)
        results_by_form.setdefault(form, []).append(prefers_male)

    all_arr = np.array(all_results)
    per_form = {form: float(np.mean(vals)) for form, vals in results_by_form.items()}

    return {
        "male_pref_rate": float(np.mean(all_arr)) if len(all_arr) > 0 else 0.5,
        "per_pronoun_form": per_form,
        "n_pairs": len(all_arr),
        "_per_example": all_arr,
    }


# ═══════════════════════════════════════════════════════════════
# WINOBIAS — coreference accuracy: pro vs anti stereotypical
# ═══════════════════════════════════════════════════════════════

_winobias_cache = {}

def _load_winobias():
    if not _winobias_cache:
        from datasets import load_dataset
        for config in ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]:
            ds = load_dataset("wino_bias", config, split="test")
            sentences = []
            for ex in ds:
                sent = " ".join(ex["tokens"])
                sentences.append(sent)
            _winobias_cache[config] = sentences
    return _winobias_cache


def _find_pronoun_and_prefix(sentence, model):
    """Find gendered pronoun in WinoBias sentence, return (prefix_tokens, male_id, female_id) or None."""
    words = sentence.split()
    he_id = _get_token_id(model, " he")
    she_id = _get_token_id(model, " she")
    him_id = _get_token_id(model, " him")
    her_id = _get_token_id(model, " her")
    his_id = _get_token_id(model, " his")

    gendered = {"he": (he_id, she_id), "she": (she_id, he_id),
                "him": (him_id, her_id), "her": (her_id, him_id),
                "his": (his_id, her_id)}

    for i, w in enumerate(words):
        clean = w.lower().rstrip(".,!?;:")
        if clean in gendered and i > 0:
            prefix = " ".join(words[:i])
            correct_id, incorrect_id = gendered[clean]
            return prefix, correct_id, incorrect_id
    return None


def eval_winobias(model, hooks=None):
    """
    WinoBias: for each sentence, find the pronoun position and check if
    P(correct_pronoun | prefix) > P(incorrect_pronoun | prefix).

    Returns pro/anti accuracy for type1 and type2, plus gap.
    """
    data = _load_winobias()
    results = {}

    for config, sentences in data.items():
        correct_count = 0
        total = 0
        per_ex = []

        for sent in sentences:
            parsed = _find_pronoun_and_prefix(sent, model)
            if parsed is None:
                continue
            prefix, correct_id, incorrect_id = parsed

            tokens = model.to_tokens(prefix)
            with torch.no_grad():
                logits = model.run_with_hooks(tokens, fwd_hooks=hooks) if hooks else model(tokens)
            probs = torch.softmax(logits[0, -1, :], dim=0)

            is_correct = int(probs[correct_id].item() > probs[incorrect_id].item())
            correct_count += is_correct
            total += 1
            per_ex.append(is_correct)

        acc = correct_count / total if total > 0 else 0.0
        results[config] = {"accuracy": float(acc), "n": total, "_per_example": np.array(per_ex)}

    # Compute gaps
    t1_pro = results.get("type1_pro", {}).get("accuracy", 0)
    t1_anti = results.get("type1_anti", {}).get("accuracy", 0)
    t2_pro = results.get("type2_pro", {}).get("accuracy", 0)
    t2_anti = results.get("type2_anti", {}).get("accuracy", 0)

    return {
        "type1_pro_acc": t1_pro,
        "type1_anti_acc": t1_anti,
        "type1_gap": t1_pro - t1_anti,
        "type2_pro_acc": t2_pro,
        "type2_anti_acc": t2_anti,
        "type2_gap": t2_pro - t2_anti,
        "overall_acc": float(np.mean([t1_pro, t1_anti, t2_pro, t2_anti])),
        "_details": results,
    }


# ═══════════════════════════════════════════════════════════════
# CROWS-PAIRS (full dataset from GitHub)
# ═══════════════════════════════════════════════════════════════

_crows_cache = None

def _load_crows_pairs():
    global _crows_cache
    if _crows_cache is None:
        import pandas as pd
        url = "https://raw.githubusercontent.com/nyu-mll/crows-pairs/master/data/crows_pairs_anonymized.csv"
        df = pd.read_csv(url)
        gender_df = df[df["bias_type"] == "gender"]
        _crows_cache = list(zip(
            gender_df["sent_more"].tolist(),
            gender_df["sent_less"].tolist()
        ))
    return _crows_cache


def eval_crows_pairs(model, hooks=None):
    """CrowS-Pairs gender: fraction preferring stereotypical sentence. Returns score + per-pair."""
    pairs = _load_crows_pairs()
    per_pair = []
    for stereo, anti in pairs:
        pll_s = pseudo_log_likelihood(model, stereo, hooks)
        pll_a = pseudo_log_likelihood(model, anti, hooks)
        per_pair.append(1 if pll_s > pll_a else 0)
    arr = np.array(per_pair)
    return {
        "stereotype_score": float(np.mean(arr)),
        "n_pairs": len(arr),
        "_per_pair": arr,
    }


# ═══════════════════════════════════════════════════════════════
# GAP (real-world pronoun coreference from Wikipedia)
# ═══════════════════════════════════════════════════════════════

_gap_cache = None

def _load_gap():
    global _gap_cache
    if _gap_cache is None:
        import pandas as pd
        url = "https://raw.githubusercontent.com/google-research-datasets/gap-coreference/master/gap-test.tsv"
        _gap_cache = pd.read_csv(url, sep="\t")
    return _gap_cache


def eval_gap(model, hooks=None):
    """
    GAP: For each example, compute sentence PLL with pronoun replaced by
    candidate A vs candidate B name. Check if the higher-PLL replacement
    matches the ground truth coreference.

    Returns masculine/feminine accuracy splits.
    """
    df = _load_gap()
    results = {"M": [], "F": []}

    for _, row in df.iterrows():
        text = row["Text"]
        pronoun = row["Pronoun"]
        pron_offset = int(row["Pronoun-offset"])
        a_name = row["A"]
        b_name = row["B"]
        a_coref = row["A-coref"]
        b_coref = row["B-coref"]

        # Replace pronoun with each candidate
        before = text[:pron_offset]
        after = text[pron_offset + len(pronoun):]
        text_a = before + a_name + after
        text_b = before + b_name + after

        pll_a = pseudo_log_likelihood(model, text_a, hooks)
        pll_b = pseudo_log_likelihood(model, text_b, hooks)

        model_picks_a = pll_a > pll_b

        # Determine correctness
        if a_coref and not b_coref:
            correct = 1 if model_picks_a else 0
        elif b_coref and not a_coref:
            correct = 1 if not model_picks_a else 0
        else:
            continue  # Neither or both — skip

        # Gender of the pronoun
        gender = "M" if pronoun.lower() in ("he", "him", "his", "himself") else "F"
        results[gender].append(correct)

    m_arr = np.array(results["M"]) if results["M"] else np.array([])
    f_arr = np.array(results["F"]) if results["F"] else np.array([])

    return {
        "masculine_acc": float(np.mean(m_arr)) if len(m_arr) > 0 else 0.0,
        "feminine_acc": float(np.mean(f_arr)) if len(f_arr) > 0 else 0.0,
        "overall_acc": float(np.mean(np.concatenate([m_arr, f_arr]))) if len(m_arr) + len(f_arr) > 0 else 0.0,
        "n_masculine": len(m_arr),
        "n_feminine": len(f_arr),
        "_m_results": m_arr,
        "_f_results": f_arr,
    }


# ═══════════════════════════════════════════════════════════════
# BOOTSTRAP
# ═══════════════════════════════════════════════════════════════

def bootstrap_ci(data, n_boot=10000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    if len(data) == 0:
        return (0.0, 0.0)
    means = np.array([np.mean(np.random.choice(data, size=len(data), replace=True))
                       for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (float(np.percentile(means, 100 * alpha)),
            float(np.percentile(means, 100 * (1 - alpha))))


def add_cis(result_dict, key, data_key, n_boot=10000):
    """Add bootstrap CI to result dict for a given metric."""
    if data_key in result_dict and len(result_dict[data_key]) > 0:
        lo, hi = bootstrap_ci(result_dict[data_key], n_boot)
        result_dict[key + "_ci"] = [lo, hi]


# ═══════════════════════════════════════════════════════════════
# MASTER EVALUATION FUNCTION
# ═══════════════════════════════════════════════════════════════

def full_eval(model, hooks=None, split="dev", capability="full", n_boot=10000, verbose=True):
    """
    Run complete evaluation suite.

    Args:
        model: HookedTransformer model
        hooks: list of (hook_name, hook_fn) or None for baseline
        split: "discovery", "dev", or "test"
        capability: "full" (all benchmarks), "light" (PPL + Winogender only),
                    or False/None (bias metrics only)
        n_boot: bootstrap iterations
        verbose: print progress

    Returns:
        dict with all metrics and bootstrap CIs
    """
    results = {}

    # ── Bias metrics ──
    if verbose:
        print("  [bias] Evaluating on %s split..." % split)
    prompts = get_prompts(split)
    bias = eval_bias(model, prompts, hooks)
    results["signed_bias"] = bias["signed_bias"]
    results["abs_bias"] = bias["abs_bias"]
    results["total_gender_mass"] = bias["total_gender_mass"]
    results["stereotype_preference"] = bias["stereotype_preference"]
    results["n_prompts"] = len(prompts)

    # Bootstrap CIs for bias
    lo, hi = bootstrap_ci(bias["_signed_scores"], n_boot)
    results["signed_bias_ci"] = [lo, hi]
    lo, hi = bootstrap_ci(bias["_abs_scores"], n_boot)
    results["abs_bias_ci"] = [lo, hi]

    if not capability:
        return results

    # ── WikiText PPL ──
    n_wt = 1000 if capability == "full" else 200
    if verbose:
        print("  [capability] WikiText-103 PPL (%d sentences)..." % n_wt)
    ppl_arr = eval_wikitext_ppl(model, hooks, n_sentences=n_wt)
    results["wikitext_ppl"] = float(np.mean(ppl_arr))
    results["wikitext_ppl_median"] = float(np.median(ppl_arr))
    lo, hi = bootstrap_ci(ppl_arr, n_boot)
    results["wikitext_ppl_ci"] = [lo, hi]

    # ── Winogender ──
    if verbose:
        print("  [coref] Winogender...")
    wg = eval_winogender(model, hooks)
    results["winogender_male_pref"] = wg["male_pref_rate"]
    results["winogender_per_form"] = wg["per_pronoun_form"]
    results["winogender_n"] = wg["n_pairs"]
    if len(wg["_per_example"]) > 0:
        lo, hi = bootstrap_ci(wg["_per_example"], n_boot)
        results["winogender_male_pref_ci"] = [lo, hi]

    if capability != "full":
        return results

    # ── LAMBADA ──
    if verbose:
        print("  [capability] LAMBADA (500 examples)...")
    lam_arr = eval_lambada(model, hooks, n_examples=500)
    results["lambada_acc"] = float(np.mean(lam_arr))
    lo, hi = bootstrap_ci(lam_arr, n_boot)
    results["lambada_acc_ci"] = [lo, hi]

    # ── BLiMP ──
    if verbose:
        print("  [capability] BLiMP minimal pairs...")
    blimp_acc, blimp_per = eval_blimp(model, hooks)
    results["blimp"] = {k: v for k, v in blimp_acc.items()}
    for config, arr in blimp_per.items():
        lo, hi = bootstrap_ci(arr, n_boot)
        results["blimp"][config + "_ci"] = [lo, hi]

    # ── WinoBias ──
    if verbose:
        print("  [coref] WinoBias (pro/anti stereotypical)...")
    wb = eval_winobias(model, hooks)
    results["winobias_type1_pro"] = wb["type1_pro_acc"]
    results["winobias_type1_anti"] = wb["type1_anti_acc"]
    results["winobias_type1_gap"] = wb["type1_gap"]
    results["winobias_type2_pro"] = wb["type2_pro_acc"]
    results["winobias_type2_anti"] = wb["type2_anti_acc"]
    results["winobias_type2_gap"] = wb["type2_gap"]
    results["winobias_overall"] = wb["overall_acc"]
    # CIs on type1 gap
    if "_details" in wb:
        for cfg in ["type1_pro", "type1_anti", "type2_pro", "type2_anti"]:
            if cfg in wb["_details"] and len(wb["_details"][cfg]["_per_example"]) > 0:
                lo, hi = bootstrap_ci(wb["_details"][cfg]["_per_example"], n_boot)
                results["winobias_%s_ci" % cfg] = [lo, hi]

    # ── CrowS-Pairs ──
    if verbose:
        print("  [bias] CrowS-Pairs (~262 gender pairs)...")
    cp = eval_crows_pairs(model, hooks)
    results["crows_pairs_score"] = cp["stereotype_score"]
    results["crows_pairs_n"] = cp["n_pairs"]
    lo, hi = bootstrap_ci(cp["_per_pair"], n_boot)
    results["crows_pairs_ci"] = [lo, hi]

    # ── GAP ──
    if verbose:
        print("  [coref] GAP (Wikipedia pronoun coreference)...")
    gap = eval_gap(model, hooks)
    results["gap_overall"] = gap["overall_acc"]
    results["gap_masculine"] = gap["masculine_acc"]
    results["gap_feminine"] = gap["feminine_acc"]
    results["gap_n"] = gap["n_masculine"] + gap["n_feminine"]
    combined_gap = np.concatenate([gap["_m_results"], gap["_f_results"]]) if gap["n_masculine"] + gap["n_feminine"] > 0 else np.array([])
    if len(combined_gap) > 0:
        lo, hi = bootstrap_ci(combined_gap, n_boot)
        results["gap_overall_ci"] = [lo, hi]

    return results


# ═══════════════════════════════════════════════════════════════
# PRETTY PRINT
# ═══════════════════════════════════════════════════════════════

def print_results(results, label="Results"):
    """Pretty-print evaluation results."""
    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)

    def fmt_ci(key):
        ci_key = key + "_ci"
        if ci_key in results:
            return " [%.4f, %.4f]" % (results[ci_key][0], results[ci_key][1])
        return ""

    print("\n  BIAS METRICS (%d prompts):" % results.get("n_prompts", 0))
    print("    Signed bias:          %+.4f%s" % (results["signed_bias"], fmt_ci("signed_bias")))
    print("    Absolute bias:         %.4f%s" % (results["abs_bias"], fmt_ci("abs_bias")))
    print("    Total gender mass:     %.4f" % results["total_gender_mass"])
    print("    Stereotype pref (>M):  %.1f%%" % (results["stereotype_preference"] * 100))

    if "wikitext_ppl" in results:
        print("\n  CAPABILITY:")
        print("    WikiText-103 PPL:     %.2f%s" % (results["wikitext_ppl"], fmt_ci("wikitext_ppl")))
        if "lambada_acc" in results:
            print("    LAMBADA accuracy:     %.1f%%%s" % (results["lambada_acc"] * 100, fmt_ci("lambada_acc")))
        if "blimp" in results:
            print("    BLiMP mean:           %.1f%%" % (results["blimp"].get("mean", 0) * 100))
            for k, v in results["blimp"].items():
                if not k.endswith("_ci") and k != "mean":
                    print("      %-30s %.1f%%" % (k, v * 100))

    if "winogender_male_pref" in results:
        print("\n  COREFERENCE:")
        print("    Winogender male pref: %.1f%% (%d pairs)%s" % (
            results["winogender_male_pref"] * 100, results["winogender_n"], fmt_ci("winogender_male_pref")))
        if "winogender_per_form" in results:
            for form, rate in results["winogender_per_form"].items():
                print("      %-20s %.1f%% male" % (form, rate * 100))
        if "winobias_type1_pro" in results:
            print("    WinoBias Type1 pro:   %.1f%%" % (results["winobias_type1_pro"] * 100))
            print("    WinoBias Type1 anti:  %.1f%%" % (results["winobias_type1_anti"] * 100))
            print("    WinoBias Type1 gap:   %.1f pp" % (results["winobias_type1_gap"] * 100))
            print("    WinoBias Type2 pro:   %.1f%%" % (results["winobias_type2_pro"] * 100))
            print("    WinoBias Type2 anti:  %.1f%%" % (results["winobias_type2_anti"] * 100))
            print("    WinoBias Type2 gap:   %.1f pp" % (results["winobias_type2_gap"] * 100))
        if "gap_overall" in results:
            print("    GAP overall:          %.1f%% (M: %.1f%%, F: %.1f%%)" % (
                results["gap_overall"] * 100, results["gap_masculine"] * 100, results["gap_feminine"] * 100))

    if "crows_pairs_score" in results:
        print("\n  BIAS BENCHMARKS:")
        print("    CrowS-Pairs score:    %.1f%% (%d pairs)%s" % (
            results["crows_pairs_score"] * 100, results["crows_pairs_n"], fmt_ci("crows_pairs_score")))

    print()


def results_to_json(results):
    """Convert results to JSON-serializable dict (strip numpy arrays)."""
    clean = {}
    for k, v in results.items():
        if k.startswith("_"):
            continue
        if isinstance(v, np.ndarray):
            continue
        if isinstance(v, dict):
            clean[k] = {kk: vv for kk, vv in v.items()
                        if not isinstance(vv, np.ndarray) and not kk.startswith("_")}
        elif isinstance(v, (np.floating, np.integer)):
            clean[k] = float(v)
        else:
            clean[k] = v
    return clean
