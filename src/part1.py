"""Part 1 — How unequal is tokenisation?  (Tasks 1.1–1.3)"""
import io
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import DATA, PART1_LANGS, RESULTS, load_flores, tok

OUT = RESULTS / "part1"
OUT.mkdir(parents=True, exist_ok=True)
WIKI_SNAPSHOT = DATA / "list_of_wikipedias_2026-10-13.html"  # consulted 13 Oct 2026

dev = {code: load_flores(code, "dev") for code in PART1_LANGS}
assert all((d["id"] == dev["eng_Latn"]["id"]).all() for d in dev.values())  # n-way parallel


# ---------------------------------------------------------------- Task 1.1
def show_pieces(code, i=0):
    enc = tok(dev[code]["text"][i], add_special_tokens=False)
    pieces = tok.convert_ids_to_tokens(enc["input_ids"])
    lines = [f"=== {code} ===", dev[code]["text"][i]]
    lines += [f"{w!s:>4}  {p}" for p, w in zip(pieces, enc.word_ids())]
    return "\n".join(lines)


task11 = "\n\n".join(show_pieces(c) for c in
                     ["eng_Latn", "fra_Latn", "fin_Latn", "cmn_Hans", "tha_Thai", "wol_Latn", "amh_Ethi"])
(OUT / "task1_1_pieces.txt").write_text(task11)


# ---------------------------------------------------------------- Task 1.2
def fragmentation(sentences):
    """Pieces, words, words split in >= 2 pieces, [UNK] pieces, characters swallowed by [UNK]."""
    enc = tok(list(sentences), add_special_tokens=False, return_offsets_mapping=True)
    n_pieces = n_words = n_cont = n_unk = unk_chars = 0
    for i, ids in enumerate(enc["input_ids"]):
        per_word = Counter(enc.word_ids(i))          # word index -> number of pieces
        n_pieces += len(ids)
        n_words += len(per_word)
        n_cont += sum(c >= 2 for c in per_word.values())
        for tid, (a, b) in zip(ids, enc["offset_mapping"][i]):
            if tid == tok.unk_token_id:
                n_unk += 1
                unk_chars += b - a
    n_chars = sum(len("".join(s.split())) for s in sentences)  # non-space characters
    return dict(pieces=n_pieces, words=n_words, cont=n_cont, unk=n_unk,
                unk_char_share=unk_chars / n_chars)


def long_words(sentences, limit=tok.backend_tokenizer.model.max_input_chars_per_word):
    """Pre-tokenised 'words' longer than the WordPiece limit (they become a single [UNK])."""
    bt = tok.backend_tokenizer
    words = [w for s in sentences
             for w, _ in bt.pre_tokenizer.pre_tokenize_str(bt.normalizer.normalize_str(s))]
    return sum(len(w) > limit for w in words), max(len(w) for w in words)


rows = []
for code, (name, script, seen, wiki) in PART1_LANGS.items():
    f = fragmentation(dev[code]["text"])
    n_long, longest = long_words(dev[code]["text"])
    rows.append(dict(code=code, language=name, script=script, in_mbert=seen, wiki=wiki,
                     pieces=f["pieces"], words=f["words"],
                     fertility=f["pieces"] / f["words"], kappa=f["cont"] / f["words"],
                     unk_rate=f["unk"] / f["pieces"], unk_char_share=f["unk_char_share"],
                     long_words=n_long, longest_word=longest))
df = pd.DataFrame(rows)
df["premium"] = df["pieces"] / df.loc[df.code == "eng_Latn", "pieces"].item()

# Resource proxy W_L: number of articles, from meta.wikimedia.org/wiki/List_of_Wikipedias
wl = pd.read_html(io.StringIO(WIKI_SNAPSHOT.read_text()))[0]
wl = wl.rename(columns={"Wiki [2]": "wiki"})[["wiki", "Articles"]]
df = df.merge(wl.rename(columns={"Articles": "W"}), on="wiki", how="left")
assert df["W"].notna().all(), df[df["W"].isna()]
df["log10_W"] = np.log10(df["W"].astype(float))

df = df.sort_values("premium").reset_index(drop=True)
df.to_csv(OUT / "table_fragmentation.csv", index=False)

# Spearman over languages seen by mBERT (+ robustness checks)
seen = df[df.in_mbert]
rho, pval = spearmanr(seen["premium"], seen["W"])
no_bots = seen[~seen.code.isin(["ceb_Latn", "war_Latn"])]
rho_nb, pval_nb = spearmanr(no_bots["premium"], no_bots["W"])
rho_all, pval_all = spearmanr(df["premium"], df["W"])
stats = (f"Spearman(premium, W), seen by mBERT (n={len(seen)}): rho={rho:.3f}, p={pval:.4f}\n"
         f"  without ceb/war (n={len(no_bots)}): rho={rho_nb:.3f}, p={pval_nb:.4f}\n"
         f"  all 23 languages: rho={rho_all:.3f}, p={pval_all:.4f}\n"
         f"max_input_chars_per_word = {tok.backend_tokenizer.model.max_input_chars_per_word}\n")
(OUT / "spearman.txt").write_text(stats)

# Plot premium vs log10 W
fig, ax = plt.subplots(figsize=(7, 4.8))
for flag, marker, label in [(True, "o", "in mBERT pre-training"), (False, "x", "not in mBERT")]:
    sub = df[df.in_mbert == flag]
    ax.scatter(sub["log10_W"], sub["premium"], marker=marker, s=40, label=label,
               color="C0" if flag else "C3")
nudge = {"heb_Hebr": (-20, -12), "fin_Latn": (-8, 6), "kor_Hang": (-14, -12), "arb_Arab": (-38, 4),
         "jpn_Jpan": (4, 5), "cmn_Hans": (-6, -13), "rus_Cyrl": (4, 4)}
for _, r in df.iterrows():
    ax.annotate(r.code, (r.log10_W, r.premium), fontsize=7, xytext=nudge.get(r.code, (3, 3)),
                textcoords="offset points")
ax.axhline(1, color="grey", lw=0.8, ls="--")
ax.set_xlabel(r"$\log_{10} W_L$ (Wikipedia articles, 13 Oct 2026)")
ax.set_ylabel(r"tokenisation premium $\pi_L$ (vs English)")
ax.set_title(rf"Premium vs Wikipedia size — Spearman $\rho$={rho:.2f} (seen langs, n={len(seen)})")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "premium_vs_wiki.png", dpi=200)

with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(df[["code", "in_mbert", "premium", "fertility", "kappa", "unk_rate", "unk_char_share",
              "long_words", "longest_word", "W"]].round(3).to_string())
print(stats)
