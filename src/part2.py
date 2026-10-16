"""Part 2 — Do languages share a subspace?  (Tasks 2.1–2.6, run on raw and standardised representations)

Requires the cached representations of part2_extract.py.  Everything is fitted on dev, evaluated on devtest.
"""
import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import DATA, LANGS, RESULTS, SCRIPT, load_flores

OUT = RESULTS / "part2"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 0
N_LAYERS, D = 13, 768
SHORT = {c: c[:3] for c in LANGS}
summary = {}  # numbers quoted in the report


def load_reps():
    return {s: {c: np.load(DATA / "reps" / f"{c}.{s}.npy").astype(np.float64) for c in LANGS}
            for s in ["dev", "devtest"]}


R = {"raw": load_reps()}


# ---------------------------------------------------------------- Task 2.1 — inspect
def mean_random_cosine(X, n_pairs=1000, rng=None):
    i = rng.integers(0, len(X), n_pairs)
    j = (i + rng.integers(1, len(X), n_pairs)) % len(X)  # j != i: distinct sentences
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    return float((U[i] * U[j]).sum(1).mean())


def task_2_1():
    rng = np.random.default_rng(SEED)
    rows = []
    for c in LANGS:
        for l in range(N_LAYERS):
            X = R["raw"]["dev"][c][l]
            rows.append(dict(lang=c, layer=l, norm=np.linalg.norm(X, axis=1).mean(),
                             cos=mean_random_cosine(X, rng=rng),
                             cos_centred=mean_random_cosine(X - X.mean(0), rng=rng)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t21_norm_cosine.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for c in LANGS:
        s = df[df.lang == c]
        ax[0].plot(s.layer, s.norm, marker=".", label=c)
        ax[1].plot(s.layer, s.cos, marker=".", label=c)
    ax[1].axhline(0, color="grey", lw=0.8, ls="--")
    ax[1].text(0.2, 0.03, "isotropic baseline: 0 ± 0.036", fontsize=7, color="grey")
    ax[0].set(xlabel="layer", ylabel="mean L2 norm", title="Average norm (dev)")
    ax[1].set(xlabel="layer", ylabel="mean cosine", title="Mean cosine of 1,000 random pairs (dev)", ylim=(-0.05, 1))
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "t21_norm_cosine.png", dpi=200)


# ---------------------------------------------------------------- Task 2.2 — rogue dimensions
def cc_terms(X):
    U = X / np.linalg.norm(X, axis=1, keepdims=True)
    return U.mean(0) ** 2  # CC_i = (mean u_i)^2


def task_2_2():
    rows = []
    for l in range(N_LAYERS):
        groups = {"pooled": np.concatenate([R["raw"]["dev"][c][l] for c in LANGS])}
        groups.update({c: R["raw"]["dev"][c][l] for c in LANGS})
        for g, X in groups.items():
            cc = cc_terms(X)
            top = np.argsort(cc)[::-1][:3]
            rows.append(dict(layer=l, group=g, sum_cc=cc.sum(), top3_share=cc[top].sum() / cc.sum(),
                             top3=" ".join(map(str, top)),
                             emp_cos=mean_random_cosine(X, 5000, np.random.default_rng(SEED))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t22_rogue.csv", index=False)

    # standardised version: one set of statistics per layer, pooled over the six dev sets
    R["std"] = {s: {} for s in R["raw"]}
    for l in range(N_LAYERS):
        pooled = np.concatenate([R["raw"]["dev"][c][l] for c in LANGS])
        mu, sd = pooled.mean(0), pooled.std(0)
        for s in R["raw"]:
            for c in LANGS:
                R["std"][s].setdefault(c, np.empty_like(R["raw"][s][c]))[l] = (R["raw"][s][c][l] - mu) / sd
    std_rows = []
    for l in range(N_LAYERS):
        cc = cc_terms(np.concatenate([R["std"]["dev"][c][l] for c in LANGS]))
        std_rows.append(dict(layer=l, sum_cc_std=cc.sum(), top3_share_std=np.sort(cc)[::-1][:3].sum() / cc.sum()))
    pd.DataFrame(std_rows).to_csv(OUT / "t22_rogue_std.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    p = df[df.group == "pooled"]
    ax.plot(p.layer, p.sum_cc, marker="o", label=r"$\sum_i CC_i$ (= expected cosine)")
    ax.plot(p.layer, p.top3_share, marker="s", label="share of the 3 largest terms")
    ax.plot(p.layer, pd.DataFrame(std_rows).top3_share_std, marker="^", ls="--",
            label="share of the 3 largest, standardised")
    ax.set(xlabel="layer", ylim=(0, 1), title="Rogue dimensions (pooled dev, 6 languages)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "t22_rogue.png", dpi=200)


# ---------------------------------------------------------------- helpers
def centred(variant, split, c, l):
    """Representation centred on the mean of its language, estimated on dev."""
    return R[variant][split][c][l] - R[variant]["dev"][c][l].mean(0)


def stack(variant, split, l, centre=False):
    X = np.concatenate([centred(variant, split, c, l) if centre else R[variant][split][c][l] for c in LANGS])
    y = np.repeat(np.arange(len(LANGS)), [len(R[variant][split][c][l]) for c in LANGS])
    return X, y


def probe():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


# ---------------------------------------------------------------- Task 2.3 — language identification
def task_2_3():
    rows = []
    for variant in ["raw", "std"]:
        for centre in [False, True]:
            for l in range(N_LAYERS):
                Xtr, ytr = stack(variant, "dev", l, centre)
                Xte, yte = stack(variant, "devtest", l, centre)
                acc = probe().fit(Xtr, ytr).score(Xte, yte)
                rows.append(dict(variant=variant, centred=centre, layer=l, acc=acc))
                print(f"2.3 {variant} centred={centre} layer {l}: {acc:.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t23_langid.csv", index=False)


def task_2_3_knn(k=10):
    """Extra (Q32): a non-linear, rotation-invariant classifier on centred vectors. Logistic regression is
    stuck at chance there (w = 0 is its optimum when all class means are 0), but k-NN is not."""
    from sklearn.neighbors import KNeighborsClassifier
    rows = []
    for variant in ["raw", "std"]:
        for l in range(N_LAYERS):
            Xtr, ytr = stack(variant, "dev", l, centre=True)
            Xte, yte = stack(variant, "devtest", l, centre=True)
            knn = KNeighborsClassifier(n_neighbors=k, metric="cosine").fit(Xtr, ytr)
            rows.append(dict(variant=variant, layer=l, acc=knn.score(Xte, yte)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t23_knn_centred.csv", index=False)

    # raw and standardised give identical curves (the pipeline's StandardScaler absorbs the standardisation)
    lr = pd.read_csv(OUT / "t23_langid.csv").query("variant == 'raw'")
    knn = df.query("variant == 'raw'")
    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.plot(lr[~lr.centred].layer, lr[~lr.centred].acc, marker="o", label="logistic regression, not centred")
    ax.plot(lr[lr.centred].layer, lr[lr.centred].acc, marker="o", ls="--", label="logistic regression, centred")
    ax.plot(knn.layer, knn.acc, marker="s", color="C2", label=f"{k}-NN (cosine), centred")
    ax.axhline(1 / 6, color="grey", ls=":", label="chance (1/6)")
    ax.set(xlabel="layer", ylabel="devtest accuracy", ylim=(0, 1.04), title="Language identification")
    ax.legend(fontsize=7, loc="center left")
    fig.tight_layout()
    fig.savefig(OUT / "t23_langid.png", dpi=200)


# ---------------------------------------------------------------- Task 2.4 — retrieval
def p_at_1(E, X):
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return float(((E @ X.T).argmax(1) == np.arange(len(E))).mean())  # row i's translation is row i


def task_2_4():
    rows = []
    for c in LANGS[1:]:
        for l in range(N_LAYERS):
            rows.append(dict(lang=c, layer=l,
                             raw=p_at_1(R["raw"]["devtest"]["eng_Latn"][l], R["raw"]["devtest"][c][l]),
                             centred=p_at_1(centred("raw", "devtest", "eng_Latn", l), centred("raw", "devtest", c, l)),
                             std=p_at_1(R["std"]["devtest"]["eng_Latn"][l], R["std"]["devtest"][c][l]),
                             std_centred=p_at_1(centred("std", "devtest", "eng_Latn", l),
                                                centred("std", "devtest", c, l))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t24_retrieval.csv", index=False)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharey=True)
    for ax, v, title in zip(axes, ["raw", "std", "centred", "std_centred"],
                            ["raw", "standardised (extra)", "centred per language", "standardised + centred"]):
        for c in LANGS[1:]:
            s = df[df.lang == c]
            ax.plot(s.layer, s[v], marker=".", label=f"eng→{SHORT[c]}")
        ax.axhline(1 / 1012, color="grey", ls=":", label="random (1/1012)")
        ax.set(title=title, xlabel="layer", ylim=(0, 1))
    axes[0].set_ylabel("precision at 1")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "t24_retrieval.png", dpi=200)


# ---------------------------------------------------------------- Task 2.5 — subspace overlap
def captured(X, U):
    """c(X, U) = ||Xc U||_F^2 / ||Xc||_F^2, X centred on its own mean (Eq. 1)."""
    Xc = X - X.mean(0)
    return float(np.linalg.norm(Xc @ U) ** 2 / np.linalg.norm(Xc) ** 2)


def principal(X, k):
    """k leading right singular vectors of the centred matrix, as a (d, k) orthonormal basis."""
    return np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:k].T


def random_basis(k, rng):
    return np.linalg.qr(rng.standard_normal((D, k)))[0]


def omega(XB_test, UA, UB, Urands):
    floor = np.mean([captured(XB_test, U) for U in Urands])
    return (captured(XB_test, UA) - floor) / (captured(XB_test, UB) - floor)


def task_2_5(ks=(10, 50, 100), n_rand=10):
    rng = np.random.default_rng(SEED)
    # Q41: E[c(X, U_rand)] = k / d, checked numerically
    X = R["raw"]["devtest"]["eng_Latn"][8]
    summary["q41"] = {k: dict(mean=float(np.mean([captured(X, random_basis(k, rng)) for _ in range(200)])),
                          expected=k / D) for k in ks}
    rows = []
    for variant in ["raw", "std"]:
        for l in range(N_LAYERS):
            for k in ks:
                Urands = [random_basis(k, rng) for _ in range(n_rand)]
                U = {c: principal(R[variant]["dev"][c][l], k) for c in LANGS}
                for A, B in itertools.permutations(LANGS, 2):
                    rows.append(dict(variant=variant, layer=l, k=k, A=A, B=B,
                                     same_script=SCRIPT[A] == SCRIPT[B],
                                     omega=omega(R[variant]["devtest"][B][l], U[A], U[B], Urands)))
        print(f"2.5 {variant} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t25_overlap.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    for ax, variant in zip(axes, ["raw", "std"]):
        g = df[df.variant == variant].groupby(["k", "same_script", "layer"]).omega.mean().reset_index()
        for i, k in enumerate(ks):
            for same, ls in [(True, "-"), (False, "--")]:
                s = g[(g.k == k) & (g.same_script == same)]
                ax.plot(s.layer, s.omega, ls=ls, color=f"C{i}", marker=".",
                        label=f"k={k}, {'same script' if same else 'different scripts'}")
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.axhline(1, color="grey", lw=0.8, ls=":")
        ax.set(title=f"Normalised overlap ω ({variant})", xlabel="layer")
    axes[0].set_ylabel(r"mean $\omega_k(A\to B)$")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "t25_overlap.png", dpi=200)

    # heat maps at one middle layer, k = 50
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, variant in zip(axes, ["raw", "std"]):
        s = df[(df.variant == variant) & (df.layer == 8) & (df.k == 50)]
        M = s.pivot(index="A", columns="B", values="omega").reindex(index=LANGS, columns=LANGS)
        im = ax.imshow(M.values, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(6), [SHORT[c] for c in LANGS])
        ax.set_yticks(range(6), [SHORT[c] for c in LANGS])
        for i, j in itertools.product(range(6), range(6)):
            if i != j:
                ax.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if M.values[i, j] < 0.6 else "black")
        ax.set(title=f"ω50(A→B), layer 8, {variant}", xlabel="B (evaluated)")
    axes[0].set_ylabel("A (subspace)")
    fig.subplots_adjust(wspace=0.15)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(OUT / "t25_heatmap_layer8.png", dpi=200, bbox_inches="tight")

    # Q43: dependence on k at one layer
    l, rows_k = 8, []
    for k in [1, 2, 5, 10, 25, 50, 100, 200, 400, 600, 750]:
        Urands = [random_basis(k, rng) for _ in range(n_rand)]
        U = {c: principal(R["raw"]["dev"][c][l], k) for c in LANGS}
        om = [omega(R["raw"]["devtest"][B][l], U[A], U[B], Urands) for A, B in itertools.permutations(LANGS, 2)]
        cB = np.mean([captured(R["raw"]["devtest"][B][l], U[B]) for B in LANGS])
        rows_k.append(dict(k=k, omega=float(np.mean(om)), ceiling=float(cB), floor=k / D))
    pd.DataFrame(rows_k).to_csv(OUT / "t25_k_dependence.csv", index=False)

    # Q42: ceiling fitted and evaluated on the same sentences (in-sample) vs dev -> devtest
    rows_c = []
    for k in ks:
        for c in LANGS:
            Xd, Xt = R["raw"]["dev"][c][l], R["raw"]["devtest"][c][l]
            rows_c.append(dict(k=k, lang=c, in_sample=captured(Xd, principal(Xd, k)),
                               held_out=captured(Xt, principal(Xd, k))))
    pd.DataFrame(rows_c).to_csv(OUT / "t25_ceiling_insample.csv", index=False)


# ---------------------------------------------------------------- Task 2.6 — same direction, same meaning?
def task_2_6():
    y = {s: load_flores("eng_Latn", s)["domain"].to_numpy() for s in ["dev", "devtest"]}  # same for every language
    base = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y["dev"]), 1)), y["dev"])
    summary["t26_majority"] = dict(cls=str(base.predict([[0]])[0]),
                                   acc=float(base.score(np.zeros((len(y["devtest"]), 1)), y["devtest"])))
    rows = []
    for variant in ["raw", "std"]:
        for centre in [False, True]:
            for l in range(N_LAYERS):
                get = (lambda s, c: centred(variant, s, c, l)) if centre else (lambda s, c: R[variant][s][c][l])
                for A in LANGS:
                    clf = probe().fit(get("dev", A), y["dev"])
                    for B in LANGS:
                        rows.append(dict(variant=variant, centred=centre, layer=l, A=A, B=B,
                                         acc=clf.score(get("devtest", B), y["devtest"])))
            print(f"2.6 {variant} centred={centre} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "t26_domain_transfer.csv", index=False)

    # raw only: the standardised curves are identical (StandardScaler in the probe), and so is the
    # within-language accuracy with or without centring (a shift of both splits is absorbed by the intercept)
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    s = df[df.variant == "raw"]
    within = s[(s.A == s.B) & ~s.centred].groupby("layer").acc.mean()
    ax.plot(within.index, within, color="C0", marker="o", label="within language")
    mya = (s.A == "mya_Mymr") | (s.B == "mya_Mymr")
    groups = [("Latin ↔ Latin (same script)", (s.A.map(SCRIPT) == s.B.map(SCRIPT)) & (s.A != s.B), "C1"),
              ("different scripts, without Burmese", (s.A.map(SCRIPT) != s.B.map(SCRIPT)) & ~mya, "C3"),
              ("pairs with Burmese", (s.A != s.B) & mya, "C4")]
    for name, mask, color in groups:
        for centre, ls in [(False, "-"), (True, "--")]:
            t = s[mask & (s.centred == centre)].groupby("layer").acc.mean()
            ax.plot(t.index, t, ls=ls, color=color, marker=".",
                    label=f"transfer, {name}, {'centred' if centre else 'not centred'}")
    ax.axhline(summary["t26_majority"]["acc"], color="grey", ls=":", label="majority class")
    ax.set(title="Domain probe: within-language vs transfer accuracy (devtest)", xlabel="layer",
           ylabel="accuracy", ylim=(0.25, 0.8))
    ax.legend(fontsize=6.5, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "t26_domain_transfer.png", dpi=200)


def extras():
    """Numbers quoted in the report: rogue dimensions (Q46) and the overlap/transfer quadrants (Q53)."""
    # Q46: do the dominant CC dimensions carry the mean or the (centred) variance?
    rows = []
    for l in range(N_LAYERS):
        X = {c: R["raw"]["dev"][c][l] for c in LANGS}
        P = np.concatenate(list(X.values()))
        rogue = np.argsort(cc_terms(P))[::-1][:3]
        mean_share = (P.mean(0)[rogue] ** 2).sum() / (P.mean(0) ** 2).sum()
        var_share = np.mean([((x - x.mean(0))[:, rogue] ** 2).sum() / ((x - x.mean(0)) ** 2).sum() for x in X.values()])
        rows.append(dict(layer=l, dims=" ".join(map(str, rogue)), share_of_squared_mean=mean_share,
                         share_of_centred_variance=var_share, uniform=3 / D))
    pd.DataFrame(rows).to_csv(OUT / "extra_rogue_mean_vs_variance.csv", index=False)

    # Q53: overlap omega_50 vs normalised transfer at layer 8
    base = summary["t26_majority"]["acc"]
    o = pd.read_csv(OUT / "t25_overlap.csv").query("variant == 'raw' and k == 50 and layer == 8")
    d = pd.read_csv(OUT / "t26_domain_transfer.csv").query("variant == 'raw' and layer == 8")
    rows = []
    for cen in [False, True]:
        s = d[d.centred == cen]
        within = s[s.A == s.B].set_index("B").acc
        for _, r in s[s.A != s.B].iterrows():
            rows.append(dict(centred=cen, A=r.A, B=r.B, transfer=r.acc, within_B=within[r.B],
                             omega50=o[(o.A == r.A) & (o.B == r.B)].omega.item(),
                             ratio=(r.acc - base) / (within[r.B] - base)))
    pd.DataFrame(rows).to_csv(OUT / "t26_quadrants_layer8.csv", index=False)

    # Q48-49: within-language vs transfer accuracy at three layers
    d = pd.read_csv(OUT / "t26_domain_transfer.csv").query("variant == 'raw'")
    mya = (d.A == "mya_Mymr") | (d.B == "mya_Mymr")
    rows = []
    for l in [0, 8, 12]:
        s = d.layer == l
        rows.append(dict(layer=l, within=d[s & (d.A == d.B) & ~d.centred].acc.mean(),
                         **{f"{name}_{'centred' if cen else 'raw'}": d[s & m & (d.A != d.B) & (d.centred == cen)].acc.mean()
                            for name, m in [("transfer_hr", ~mya), ("transfer_mya", mya)] for cen in [False, True]}))
    pd.DataFrame(rows).to_csv(OUT / "t26_summary.csv", index=False)


if __name__ == "__main__":
    task_2_1()
    task_2_2()
    task_2_3()
    task_2_3_knn()
    task_2_4()
    task_2_5()
    task_2_6()
    extras()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
