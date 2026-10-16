"""Task 2.1 (Q14) — mean-pooled sentence representations of all 13 layers, cached in data/reps/.

encode() and sentence_reps() are the functions of the handout. The only addition is that
sentences are processed in order of length (and put back in their original order), which
reduces padding; padded positions are masked out, so the result does not depend on it.
"""
import time

import numpy as np
import torch

from common import DATA, LANGS, load_flores, load_model, tok

model = load_model()
torch.manual_seed(0)
OUT = DATA / "reps"
OUT.mkdir(parents=True, exist_ok=True)


def encode(sentences, max_length=256):
    enc = tok(sentences, padding=True, truncation=True, max_length=max_length,
              return_tensors="pt", return_special_tokens_mask=True)
    special = enc.pop("special_tokens_mask").bool()
    keep = enc["attention_mask"].bool() & ~special  # (B, T): real word pieces
    return enc, keep


def sentence_reps(sentences, batch_size=32):
    """Mean-pooled hidden states of all layers: array of shape (13, N, 768)."""
    order = np.argsort([len(tok.tokenize(s)) for s in sentences])
    chunks = []
    for i in range(0, len(sentences), batch_size):
        enc, keep = encode([sentences[j] for j in order[i:i + batch_size]])
        hs = model.bert(**enc, output_hidden_states=True).hidden_states
        H = torch.stack(hs)  # (13, B, T, 768)
        w = keep.unsqueeze(-1).float()  # (B, T, 1)
        chunks.append((H * w).sum(2) / w.sum(1))  # (13, B, 768)
    reps = torch.cat(chunks, dim=1).numpy()
    out = np.empty_like(reps)
    out[:, order] = reps  # back to the original (sentence id) order
    return out


if __name__ == "__main__":
    for code in LANGS:
        for split in ["dev", "devtest"]:
            f = OUT / f"{code}.{split}.npy"
            if f.exists():
                continue
            t = time.time()
            X = sentence_reps(list(load_flores(code, split)["text"]))
            np.save(f, X.astype(np.float32))
            print(f"{code} {split} {X.shape} {time.time() - t:.0f}s", flush=True)
