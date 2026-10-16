"""Part 3 — Language-specific neurons: re-implementing Tang et al. (2024) on mBERT  (Tasks 3.1–3.4)

FFNHooks, masked_batches and mlm_loss are the handout's code. Activation counts are accumulated
separately on the two halves of dev (to measure the stability of the selection) and summed.
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from common import DATA, LANGS, RESULTS, load_flores, tok
from part2_extract import encode, model

OUT = RESULTS / "part3"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = DATA / "part3"
CACHE.mkdir(parents=True, exist_ok=True)
N_LAYERS, N_FF = 12, model.config.intermediate_size
SHORT = {c: c[:3] for c in LANGS}
summary = {}


# ---------------------------------------------------------------- handout code
class FFNHooks:
    """Counts positive GELU activations and/or zeroes selected FFN neurons."""
    def __init__(self, model, ablate=None):
        self.layers = model.bert.encoder.layer
        self.counts = torch.zeros(len(self.layers), model.config.intermediate_size)
        self.n_tokens = 0
        self.keep = None  # (B, T) mask, set before each forward pass
        self.ablate = ablate or {}  # {layer index: LongTensor of neuron indices}

    def _hook(self, i):
        def hook(module, inputs, output):  # output: (B, T, 3072)
            if self.keep is not None:
                active = (output > 0) & self.keep.unsqueeze(-1)
                self.counts[i] += active.sum(dim=(0, 1))
            idx = self.ablate.get(i)
            if idx is not None and len(idx) > 0:
                output = output.clone()
                output[..., idx] = 0.0
            return output  # replaces the module output
        return hook

    def __enter__(self):
        self.handles = [layer.intermediate.register_forward_hook(self._hook(i))
                        for i, layer in enumerate(self.layers)]
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def activation_counts(sentences, batch_size=32):
    """Handout's activation_probabilities, returning the counts and the number of tokens."""
    with FFNHooks(model) as hooks:
        for i in range(0, len(sentences), batch_size):
            enc, keep = encode(sentences[i:i + batch_size])
            hooks.keep, hooks.n_tokens = keep, hooks.n_tokens + int(keep.sum())
            model.bert(**enc)
    return hooks.counts.numpy(), hooks.n_tokens


def masked_batches(sentences, p=0.15, seed=0, batch_size=32):
    """Masks drawn once: the same positions are masked in every condition."""
    g = torch.Generator().manual_seed(seed)
    batches = []
    for i in range(0, len(sentences), batch_size):
        enc, keep = encode(sentences[i:i + batch_size])
        m = (torch.rand(keep.shape, generator=g) < p) & keep
        labels = enc["input_ids"].masked_fill(~m, -100)
        enc["input_ids"] = enc["input_ids"].masked_fill(m, tok.mask_token_id)
        batches.append((enc, labels))
    return batches


def mlm_loss(batches, ablate=None):
    """Mean cross-entropy (nats) on masked positions, optionally with ablation."""
    total, n = 0.0, 0
    with FFNHooks(model, ablate=ablate):
        for enc, labels in batches:
            h = model.bert(**enc).last_hidden_state  # (B, T, 768)
            m = labels != -100  # masked positions only
            logits = model.cls(h[m])  # (n_masked, vocab)
            total += F.cross_entropy(logits, labels[m], reduction="sum").item()
            n += int(m.sum())
    return total / n


# ---------------------------------------------------------------- activation probabilities (cached)
def load_probs():
    """P[c] (12, 3072) on dev, plus the same on each half of dev (for stability checks)."""
    f = CACHE / "activation_counts.npz"
    if not f.exists():
        arrays = {}
        for c in LANGS:
            sents = list(load_flores(c, "dev")["text"])
            for h, part in enumerate([sents[:len(sents) // 2], sents[len(sents) // 2:]]):
                arrays[f"{c}.counts{h}"], arrays[f"{c}.n{h}"] = activation_counts(part)
            print("counted", c, flush=True)
        np.savez(f, **arrays)
    z = np.load(f)
    P = {c: (z[f"{c}.counts0"] + z[f"{c}.counts1"]) / (z[f"{c}.n0"] + z[f"{c}.n1"]) for c in LANGS}
    halves = [{c: z[f"{c}.counts{h}"] / z[f"{c}.n{h}"] for c in LANGS} for h in (0, 1)]
    ntok = {c: int(z[f"{c}.n0"] + z[f"{c}.n1"]) for c in LANGS}
    return P, halves, ntok


P, HALVES, NTOK = load_probs()


# ---------------------------------------------------------------- Task 3.1 — inspect (optional)
def task_3_1():
    A = np.stack([P[c] for c in LANGS])  # (6, 12, 3072)
    rows = [dict(layer=l, never=int((A[:, l].max(0) == 0).sum()),
                 rare=int((A[:, l].max(0) < 0.001).sum()),
                 almost_always=int((A[:, l].min(0) > 0.99).sum()),
                 median_p=float(np.median(A[:, l]))) for l in range(N_LAYERS)]
    pd.DataFrame(rows).to_csv(OUT / "t31_counts.csv", index=False)
    fig, axes = plt.subplots(3, 4, figsize=(12, 6.5), sharex=True)
    for l, ax in enumerate(axes.ravel()):
        ax.hist(A[:, l].ravel(), bins=50, range=(0, 1), color="C0")
        ax.set_yscale("log")
        ax.set_title(f"layer {l}", fontsize=9)
    fig.supxlabel("activation probability p (6 languages × 3072 neurons pooled)")
    fig.supylabel("count (log)")
    fig.tight_layout()
    fig.savefig(OUT / "t31_activation_hist.png", dpi=200)
    ns = np.array(list(NTOK.values()))
    summary["t31"] = dict(ntok=NTOK, se_max=dict(zip(LANGS, (0.5 / np.sqrt(ns)).round(5).tolist())),
                          se_tang=0.5 / np.sqrt(1e8))


# ---------------------------------------------------------------- Task 3.2 — LAPE selection
def lape(Pd, langs, top_rate=0.01, bar=0.95, prefilter=False):
    """LAPE of Tang et al. (2024). Returns {lang: {layer: LongTensor}}, entropies, tau, kept indices.
    prefilter=True reproduces their released code, which drops neurons with no p > tau before ranking."""
    A = np.stack([Pd[c] for c in langs]).reshape(len(langs), -1)  # (L, 12*3072)
    alive = A.max(0) > 0  # Q60: never-active neurons have no distribution
    norm = np.divide(A, A.sum(0), out=np.zeros_like(A), where=alive)
    ent = -(norm * np.log(np.where(norm > 0, norm, 1))).sum(0)
    ent[~alive] = np.inf
    tau = np.quantile(A, bar)  # 95th percentile of all activation probabilities
    if prefilter:
        ent[(A > tau).sum(0) == 0] = np.inf
    kept = np.argsort(ent)[:round(top_rate * A.shape[1])]  # 1 % lowest entropy
    assigned = A[:, kept] > tau  # (L, n_kept)
    sel = {c: {} for c in langs}
    for i, c in enumerate(langs):
        for flat in kept[assigned[i]]:
            sel[c].setdefault(int(flat // N_FF), []).append(int(flat % N_FF))
    sel = {c: {l: torch.tensor(sorted(v)) for l, v in d.items()} for c, d in sel.items()}
    return sel, ent, tau, kept, assigned


def as_set(sel_c):
    return {(l, int(j)) for l, idx in sel_c.items() for j in idx}


VARIANTS = {"handout": False, "prefilter": True}  # prefilter: released code of Tang et al.


def task_3_2():
    sels = {}
    for tag, pf in VARIANTS.items():
        sel, ent, tau, kept, assigned = lape(P, LANGS, prefilter=pf)
        sels[tag] = sel
        n_lang = assigned.sum(0)
        s = summary.setdefault("t32", {})[tag] = dict(
            n_kept=len(kept), tau=float(tau), max_lape=float(np.log(len(LANGS))), n_alive=int(np.isfinite(ent).sum()),
            per_lang={c: len(as_set(sel[c])) for c in LANGS}, assigned_to_0=int((n_lang == 0).sum()),
            assigned_to_1=int((n_lang == 1).sum()), assigned_to_2plus=int((n_lang >= 2).sum()),
            lape_max_kept=float(ent[kept].max()))
        # which languages share neurons (diagonal: neurons of the language)
        share = pd.DataFrame(0, index=LANGS, columns=LANGS)
        for col in assigned.T:
            idx = np.flatnonzero(col)
            for a in idx:
                for b in idx:
                    share.iloc[a, b] += 1
        share.to_csv(OUT / f"t32_shared_counts_{tag}.csv")
        per_layer = pd.DataFrame({c: [len(sel[c].get(l, [])) for l in range(N_LAYERS)] for c in LANGS})
        per_layer.index.name = "layer"
        per_layer.to_csv(OUT / f"t32_per_layer_{tag}.csv")
        # Q59: without German
        sel_nd, *_ = lape(P, [c for c in LANGS if c != "deu_Latn"], prefilter=pf)
        a, b = as_set(sel["fra_Latn"]), as_set(sel_nd["fra_Latn"])
        s["q59"] = dict(n_with=len(a), n_without=len(b), jaccard=len(a & b) / len(a | b))
        # stability: selection on each half of dev
        s0, *_ = lape(HALVES[0], LANGS, prefilter=pf)
        s1, *_ = lape(HALVES[1], LANGS, prefilter=pf)
        s["split_half_jaccard"] = {c: len(as_set(s0[c]) & as_set(s1[c])) / max(1, len(as_set(s0[c]) | as_set(s1[c])))
                                   for c in LANGS}
        # lowest-LAPE neurons, for the report
        pd.DataFrame([dict(layer=int(f // N_FF), neuron=int(f % N_FF), lape=float(ent[f]),
                           **{SHORT[c]: float(P[c].ravel()[f]) for c in LANGS}) for f in kept[:5]]
                     ).to_csv(OUT / f"t32_examples_{tag}.csv", index=False)
    summary["t32"]["jaccard_handout_vs_prefilter"] = {
        c: len(as_set(sels["handout"][c]) & as_set(sels["prefilter"][c])) /
        max(1, len(as_set(sels["handout"][c]) | as_set(sels["prefilter"][c]))) for c in LANGS}

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, tag, title in zip(axes, VARIANTS, ["handout procedure", "released code (τ-prefilter)"]):
        per_layer = pd.read_csv(OUT / f"t32_per_layer_{tag}.csv", index_col="layer")
        for c in LANGS:
            ax.plot(per_layer.index, per_layer[c], marker=".", label=c)
        ax.set(xlabel="layer (encoder.layer[l])", title=f"Selected neurons per layer — {title}")
    axes[0].set_ylabel("number of neurons")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "t32_per_layer.png", dpi=200)
    return sels


# ---------------------------------------------------------------- Task 3.3 — deactivation
def random_control(sel_c, assigned_any, rng):
    """Same number of neurons per layer, drawn among neurons assigned to no language."""
    out = {}
    for l, idx in sel_c.items():
        pool = np.setdiff1d(np.arange(N_FF), [j for (ll, j) in assigned_any if ll == l])
        out[l] = torch.tensor(np.sort(rng.choice(pool, size=len(idx), replace=False)))
    return out


def task_3_3(sels, n_sent=300, n_draws=3):
    f = OUT / "t33_losses.csv"
    if f.exists():
        df = pd.read_csv(f)
    else:
        batches = {c: masked_batches(list(load_flores(c, "devtest")["text"][:n_sent])) for c in LANGS}
        rows = [dict(variant="-", condition="baseline", ablated="-", draw=-1, evaluated=c, loss=mlm_loss(batches[c]))
                for c in LANGS]
        print("baseline done", flush=True)
        for tag, sel in sels.items():
            assigned_any = set().union(*(as_set(sel[c]) for c in LANGS))
            for k in LANGS:
                for c in LANGS:
                    rows.append(dict(variant=tag, condition="lape", ablated=k, draw=-1, evaluated=c,
                                     loss=mlm_loss(batches[c], sel[k])))
                for d in range(n_draws):
                    ctrl = random_control(sel[k], assigned_any, np.random.default_rng(d))
                    for c in LANGS:
                        rows.append(dict(variant=tag, condition="random", ablated=k, draw=d, evaluated=c,
                                         loss=mlm_loss(batches[c], ctrl)))
                print("ablated", tag, k, flush=True)
                pd.DataFrame(rows).to_csv(OUT / "t33_losses_partial.csv", index=False)
        df = pd.DataFrame(rows)
        df.to_csv(f, index=False)

    base = df[df.condition == "baseline"].set_index("evaluated").loss
    summary["t33_baseline"] = base.to_dict()
    for tag in VARIANTS:
        mats = {}
        for cond in ["lape", "random"]:
            s = df[(df.variant == tag) & (df.condition == cond)]
            m = s.groupby(["ablated", "evaluated"]).loss.mean().unstack().reindex(index=LANGS, columns=LANGS)
            mats[f"{cond}_abs"] = m - base
            mats[f"{cond}_rel"] = (m - base) / base * 100
            if cond == "random":  # spread over the 3 draws
                sd = s.groupby(["ablated", "evaluated"]).loss.std().unstack().reindex(index=LANGS, columns=LANGS)
                (sd / base * 100).to_csv(OUT / f"t33_{tag}_random_rel_sd.csv")
        for name, m in mats.items():
            m.to_csv(OUT / f"t33_{tag}_{name}.csv")

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        vmax = max(float(np.nanmax(mats["lape_rel"].values)), 1e-3)
        for ax, name, title in zip(axes, ["lape_rel", "random_rel"], ["LAPE neurons zeroed", "random control (mean of 3)"]):
            M = mats[name].values
            im = ax.imshow(M, cmap="Reds", vmin=0, vmax=vmax)
            ax.set_xticks(range(6), [SHORT[c] for c in LANGS])
            ax.set_yticks(range(6), [SHORT[c] for c in LANGS])
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > 0.6 * vmax else "black")
            ax.set(title=title, xlabel="evaluated language")
        axes[0].set_ylabel("deactivated language")
        fig.suptitle(f"Relative increase of the MLM loss (%) — {'handout procedure' if tag == 'handout' else 'released-code selection'}")
        fig.subplots_adjust(wspace=0.25)
        fig.colorbar(im, ax=axes, shrink=0.8)
        fig.savefig(OUT / f"t33_ablation_{tag}.png", dpi=200, bbox_inches="tight")


# ---------------------------------------------------------------- Task 3.4 — neurons and subspaces (optional)
def task_3_4(sels, n_random=2000):
    rng = np.random.default_rng(0)
    rows, cos_rows = [], []
    for l in range(N_LAYERS):
        reps = {c: np.load(DATA / "reps" / f"{c}.dev.npy", mmap_mode="r")[l + 1].astype(np.float64) for c in LANGS}
        mu = np.stack([reps[c].mean(0) for c in LANGS])
        offsets = mu - mu.mean(0)  # six centred language means, rank 5
        Q = np.linalg.svd(offsets.T, full_matrices=False)[0][:, :len(LANGS) - 1]  # (768, 5) orthonormal
        W = model.bert.encoder.layer[l].output.dense.weight.detach().numpy().astype(np.float64)  # (768, 3072)
        gain = model.bert.encoder.layer[l].output.LayerNorm.weight.detach().numpy().astype(np.float64)
        Wln = gain[:, None] * (W - W.mean(0))  # direction after LayerNorm's centring and gain (Q76)

        def r(M, j):
            return (np.linalg.norm(Q.T @ M[:, j], axis=0) ** 2) / (np.linalg.norm(M[:, j], axis=0) ** 2)

        rand = rng.choice(N_FF, n_random, replace=False)
        rows.append(dict(layer=l, variant="-", kind="random", r=r(W, rand).mean(), r_ln=r(Wln, rand).mean(), n=n_random))
        for tag, sel in sels.items():
            for c in LANGS:
                idx = sel[c].get(l)
                if idx is None or len(idx) == 0:
                    continue
                idx = idx.numpy()
                rows.append(dict(layer=l, variant=tag, kind=c, r=r(W, idx).mean(), r_ln=r(Wln, idx).mean(), n=len(idx)))
                # sign: does a neuron of language c (positive when active) push towards mu_c - mean?
                off = offsets[LANGS.index(c)]
                cs = (Wln[:, idx].T @ off) / (np.linalg.norm(Wln[:, idx], axis=0) * np.linalg.norm(off))
                cos_rows.append(dict(layer=l, variant=tag, lang=c, n=len(idx), cos_mean=cs.mean(),
                                     frac_positive=(cs > 0).mean()))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t34_r.csv", index=False)
    pd.DataFrame(cos_rows).to_csv(OUT / "t34_cos_offset.csv", index=False)
    rnd = df[df.kind == "random"].set_index("layer")
    summary["t34"] = {"expected": (len(LANGS) - 1) / 768}
    for tag in sels:
        s = df[df.variant == tag]
        summary["t34"][tag] = dict(
            n=int(s.n.sum()),
            selected_r=float(np.average(s.r, weights=s.n)), selected_r_ln=float(np.average(s.r_ln, weights=s.n)),
            random_r_matched=float(np.average(rnd.r.reindex(s.layer), weights=s.n)),
            random_r_ln_matched=float(np.average(rnd.r_ln.reindex(s.layer), weights=s.n)))


if __name__ == "__main__":
    task_3_1()
    selections = task_3_2()
    torch.save(selections, CACHE / "selections.pt")
    task_3_4(selections)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    task_3_3(selections)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
