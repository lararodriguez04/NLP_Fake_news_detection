"""
Back-translation data augmentation for PUBHEALTH dataset.

Strategy: EN -> ES -> EN using Helsinki-NLP MarianMT models (free, local, no API key needed).
Targets: 'mixture' (1434 samples) and 'unproven' (291 samples) classes.

Goal distribution after augmentation (approximate):
  true     : ~5078  (keep as-is)
  false    : ~3001  (keep as-is)
  mixture  : ~3000  (augment ~1566 samples, ~1.1x)
  unproven : ~1200  (augment ~909 samples, ~3.1x)
"""

import random
import time
import pandas as pd
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
SEED = 42
TARGET_LABELS = {"mixture", "unproven"}

# How many augmented copies to generate per original sample
# (on top of originals already present)
AUGMENT_TARGETS = {
    "mixture":  3000,   # ~1566 new samples needed  (2.1x total)
    "unproven": 1200,   # ~909 new samples needed   (4.1x total)
}

# Helsinki-NLP MarianMT model names
EN_TO_ES = "Helsinki-NLP/opus-mt-en-es"
ES_TO_EN = "Helsinki-NLP/opus-mt-es-en"

# Text fields to translate (translate BOTH so the pair stays coherent)
TEXT_FIELDS = ["text_1", "text_2", "main_text"]

# Max token budget per call (MarianMT hard limit is 512, but long text_2 may
# need chunking; we truncate at this many *words* before translation)
MAX_WORDS = 500

random.seed(SEED)


# ── Device ────────────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ── Load models ───────────────────────────────────────────────────────────────
def load_model(model_name):
    print(f"Loading {model_name} ...")
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(
        model_name,
        use_safetensors=True   
    ).to(device)
    mdl.eval()
    return tok, mdl


def translate_batch(texts, tokenizer, model, max_words=MAX_WORDS):
    """Translate a list of strings. Returns list of translated strings."""
    # Truncate very long texts so they fit within model limits
    truncated = []
    for t in texts:
        words = t.split()
        if len(words) > max_words:
            t = " ".join(words[:max_words]) + " [...]"
        truncated.append(t)

    inputs = tokenizer(
        truncated,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        translated_ids = model.generate(**inputs)

    return [tokenizer.decode(ids, skip_special_tokens=True) for ids in translated_ids]


# ── Back-translation of a single row ─────────────────────────────────────────
def back_translate_row(row, tok_en_es, mdl_en_es, tok_es_en, mdl_es_en):
    """Return a new row dict with back-translated text_1 and text_2."""
    new_row = dict(row)
    for field in TEXT_FIELDS:
        original = str(row[field])
        # EN -> ES
        es = translate_batch([original], tok_en_es, mdl_en_es)[0]
        # ES -> EN
        back = translate_batch([es], tok_es_en, mdl_es_en)[0]
        new_row[field] = back
    return new_row


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Load dataset
    print("Loading dataset...")
    train_df = pd.read_csv("/home/lnlpG08/nlp/DATA/train_reduced.csv")
    train_df = train_df.rename(columns={
        "claim":       "text_1",
        "explanation": "text_2",
    })

    print(f"Train size: {len(train_df)}")
    print("Label distribution (train):")
    print(train_df["label"].value_counts())

    # 2. Load translation models
    tok_en_es, mdl_en_es = load_model(EN_TO_ES)
    tok_es_en, mdl_es_en = load_model(ES_TO_EN)

    # 3. Augment minority classes
    augmented_rows = []

    for label in TARGET_LABELS:
        subset = train_df[train_df["label"] == label].reset_index(drop=True)
        current_count = len(subset)
        desired_total = AUGMENT_TARGETS[label]
        need = desired_total - current_count

        if need <= 0:
            print(f"[{label}] Already has {current_count} samples, no augmentation needed.")
            continue

        print(f"\n[{label}] Current: {current_count}  |  Need {need} more  -> target {desired_total}")

        # Sample WITH replacement if need > current_count
        sample_pool = subset.sample(
            n=need, replace=(need > current_count), random_state=SEED
        ).reset_index(drop=True)

        for i, row in tqdm(sample_pool.iterrows(), total=len(sample_pool), desc=f"  Augmenting [{label}]"):
            try:
                new_row = back_translate_row(
                    row.to_dict(),
                    tok_en_es, mdl_en_es,
                    tok_es_en, mdl_es_en,
                )
                new_row["augmented"] = True
                augmented_rows.append(new_row)
            except Exception as e:
                print(f"  Warning: skipping row {i} due to error: {e}")

    # 4. Combine and save
    if not augmented_rows:
        print("No rows were augmented.")
        return

    aug_df = pd.DataFrame(augmented_rows)
    train_df["augmented"] = False

    combined_df = pd.concat([train_df, aug_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    print("\n=== Final label distribution (augmented train) ===")
    print(combined_df["label"].value_counts())
    print(f"Total rows: {len(combined_df)}")

    # Save
    combined_df.to_csv("/home/lnlpG08/nlp/DATA/train_augmented_2.csv", index=False)

    print("\nSaved:")
    print("  train_augmented.csv  (original + augmented, shuffled)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")