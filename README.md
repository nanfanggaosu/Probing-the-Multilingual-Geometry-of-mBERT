# Probing the Multilingual Geometry of mBERT

Are languages represented in disjoint subspaces of a multilingual encoder? This repository tests the
hypothesis on multilingual BERT from two angles: the **geometry of the hidden states** (principal
subspaces, offsets, cross-lingual transfer) and **individual feed-forward neurons** (a re-implementation
of the LAPE method of Tang et al., 2024, with causal ablations). M2 lab, Université Paris Cité, 2026–2027.

## Main findings

- **Tokenisation (Part 1).** On 23 FLORES+ languages, the tokenisation premium decreases with Wikipedia
  size (Spearman ρ = −0.62 over the 20 languages seen by mBERT), but script, morphology and a flawed
  resource proxy (bot-generated Cebuano/Waray Wikipedias) explain much of the scatter. Piece-level [UNK]
  rates hide losses: e.g. Thai loses 7.7 % of its characters to [UNK] although only 0.1 % of its pieces are [UNK].
- **Geometry (Part 2).** In layers 6–11 the principal subspaces of different languages overlap far above a
  random floor (normalised overlap ω₅₀ ≈ 0.72–0.92), translations are retrieved at ≥ 0.95 once each language
  is centred, and a domain probe transfers across languages and scripts almost as well as within a language.
  Language identity is mainly a **mean offset**: after per-language centring a linear classifier is exactly at
  chance (w = 0 is then the optimum of the logistic loss). Rogue dimensions dominate cosine similarity but
  carry little variance, so they barely affect ω.
- **Neurons (Part 3).** Implemented literally, the selection keeps mostly dead neurons (302 of the 369
  lowest-entropy neurons are assigned to no language); the released code of Tang et al. filters by the
  activation threshold *before* ranking, which selects genuinely language-specific neurons. Both variants
  were evaluated. Selected neurons sit mostly in the last two layers; zeroing them hurts their own language
  far more than the others (except for English, whose neurons are shared with French and German), yet they
  write 90–97 % of their output into the space shared by all languages — consistent with superposition.

Low-resource language used in Parts 2–3: **Burmese (`mya_Mymr`)**.

## Set-up

- Model: `google-bert/bert-base-multilingual-cased`, revision `3f076fdb1ab68d5b2880cb87a0886f315b8146f8`.
- Data: [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) (`dev` to fit, `devtest` to
  evaluate). The dataset is **gated and must not be reposted**, so it is not included: accept its conditions
  on Hugging Face and run `huggingface_hub.login()` once; the scripts download and cache it in `data/`.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cd src
python part1.py           # Part 1 — tokenisation (< 1 min)
python part2_extract.py   # sentence representations, 13 layers × 6 languages × 2 splits (≈ 2 min, CPU)
python part2.py           # Part 2 — Tasks 2.1–2.6, raw and standardised (≈ 1.5 min)
python part3.py           # Part 3 — activation probabilities, LAPE, ablations, Task 3.4 (≈ 30 min)
```

Seeds are fixed; everything runs on a laptop CPU.

## Repository layout

| Path | Content |
|---|---|
| `src/common.py` | pinned tokenizer/model, cached FLORES+ loader, language lists |
| `src/part1.py` | premium, fertility, continued words, [UNK] rate, Spearman, plot |
| `src/part2_extract.py`, `src/part2.py` | representations; probes, retrieval, rogue dimensions, overlap ω, domain transfer |
| `src/part3.py` | FFN hooks, LAPE (both variants), stability checks, ablations with matched random controls, neurons vs. subspaces |
| `results/part{1,2,3}/` | all tables (CSV), figures (PNG) and `summary.json` files produced by the scripts |
| `data/list_of_wikipedias_2026-10-13.html` | snapshot of meta.wikimedia.org/wiki/List_of_Wikipedias used for the resource proxy |
