"""Shared set-up: pinned mBERT tokenizer and FLORES+ loader (from the lab handout)."""
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

MODEL_ID = "google-bert/bert-base-multilingual-cased"
REVISION = "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"  # pin the exact checkpoint

tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)


def load_flores(code, split):
    """One FLORES+ language variety as a DataFrame sorted by sentence id (cached as parquet)."""
    cache = DATA / "flores" / f"{code}.{split}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    df = load_dataset("openlanguagedata/flores_plus", code, split=split).to_pandas()
    df["id"] = df["id"].astype(int)
    df = df.sort_values("id").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def load_model():
    """Pinned mBERT with its MLM head, in eval mode, gradients off (imported lazily: Part 1 needs no torch)."""
    import torch
    from transformers import AutoModelForMaskedLM
    torch.set_grad_enabled(False)  # inference only: saves memory and time
    return AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=REVISION).eval()


# the six languages of Parts 2 and 3 (low-resource language chosen in Task 1.3: Burmese)
LANGS = ["eng_Latn", "fra_Latn", "deu_Latn", "rus_Cyrl", "cmn_Hans", "mya_Mymr"]
SCRIPT = {"eng_Latn": "Latn", "fra_Latn": "Latn", "deu_Latn": "Latn",
          "rus_Cyrl": "Cyrl", "cmn_Hans": "Hans", "mya_Mymr": "Mymr"}

# Table 2 of the handout: code -> (name, script, in mBERT pre-training, Wikipedia code)
PART1_LANGS = {
    "amh_Ethi": ("Amharic", "Ethiopic", False, "am"),
    "arb_Arab": ("Arabic", "Arabic", True, "ar"),
    "ceb_Latn": ("Cebuano", "Latin", True, "ceb"),
    "cmn_Hans": ("Mandarin", "Han", True, "zh"),
    "deu_Latn": ("German", "Latin", True, "de"),
    "eng_Latn": ("English", "Latin", True, "en"),
    "fin_Latn": ("Finnish", "Latin", True, "fi"),
    "fra_Latn": ("French", "Latin", True, "fr"),
    "hat_Latn": ("Haitian Creole", "Latin", True, "ht"),
    "heb_Hebr": ("Hebrew", "Hebrew", True, "he"),
    "hin_Deva": ("Hindi", "Devanagari", True, "hi"),
    "isl_Latn": ("Icelandic", "Latin", True, "is"),
    "jpn_Jpan": ("Japanese", "Kanji+kana", True, "ja"),
    "kor_Hang": ("Korean", "Hangul", True, "ko"),
    "mya_Mymr": ("Burmese", "Myanmar", True, "my"),
    "rus_Cyrl": ("Russian", "Cyrillic", True, "ru"),
    "swh_Latn": ("Swahili", "Latin", True, "sw"),
    "tgk_Cyrl": ("Tajik", "Cyrillic", True, "tg"),
    "tha_Thai": ("Thai", "Thai", True, "th"),
    "war_Latn": ("Waray", "Latin", True, "war"),
    "wol_Latn": ("Wolof", "Latin", False, "wo"),
    "yor_Latn": ("Yoruba", "Latin", True, "yo"),
    "zul_Latn": ("Zulu", "Latin", False, "zu"),
}
