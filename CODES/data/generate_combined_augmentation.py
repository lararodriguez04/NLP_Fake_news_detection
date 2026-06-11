# Genera el dataset augmentado combinando tres técnicas:
# 1. Random swap + deletion (desde train_augmented_3.csv ya generado)
# 2. Back-translation EN-ES-EN (desde train_augmented_2.csv ya generado)
# 3. Sinónimos con BERT fill-mask (generado aquí)
# Target: mixture y unproven hasta 5078. False sin augmentar.

import random
import numpy as np
import pandas as pd
import torch
from transformers import pipeline

SEED = 42
TARGET = 5078
AUGMENT_RATIO = 0.15
random.seed(SEED)

DATA_PATH = "/home/lnlpG08/nlp/DATA/"

print("Cargando datos...")
train_df = pd.read_csv(DATA_PATH + "train_reduced.csv")
train_df.dropna(subset=["claim", "main_text"], inplace=True)
train_df = train_df[train_df["label"].isin(["true", "false", "mixture", "unproven"])].copy()
train_df["augmented"] = False

print(f"Train size: {len(train_df)}")
print("\nDistribución original:")
print(train_df["label"].value_counts())

print("\nCargando ejemplos pre-generados de random swap/deletion...")
df_random = pd.read_csv(DATA_PATH + "train_augmented_3.csv")
df_random = df_random[(df_random["augmented"] == True) & (df_random["label"].isin(["mixture", "unproven"]))][["claim", "main_text", "label"]].copy()
df_random["augmented"] = True
df_random["aug_type"] = "random"
print(f"  Ejemplos random disponibles: {len(df_random)}")
print(df_random["label"].value_counts())

print("\nCargando ejemplos pre-generados de back-translation...")
df_backtrans = pd.read_csv(DATA_PATH + "train_augmented_2.csv")
df_backtrans = df_backtrans[(df_backtrans["augmented"] == True) & (df_backtrans["label"].isin(["mixture", "unproven"]))].rename(columns={"text_1": "claim"})[["claim", "main_text", "label"]].copy()
df_backtrans["augmented"] = True
df_backtrans["aug_type"] = "backtrans"
print(f"  Ejemplos backtrans disponibles: {len(df_backtrans)}")
print(df_backtrans["label"].value_counts())

print("\nCargando BERT fill-mask para sinónimos...")
fill_mask = pipeline("fill-mask", model="bert-base-uncased", device=0)

def augment_synonyms(text):
    words = text.split()
    n = max(1, int(len(words) * AUGMENT_RATIO))
    indices = random.sample(range(len(words)), min(n, len(words)))
    new_words = words.copy()
    for idx in indices:
        masked = words.copy()
        masked[idx] = "[MASK]"
        masked_text = " ".join(masked)
        if len(masked_text.split()) > 512:
            continue
        try:
            preds = fill_mask(masked_text[:512])
            replacement = preds[0]["token_str"].strip()
            if replacement and replacement != words[idx]:
                new_words[idx] = replacement
        except:
            pass
    return " ".join(new_words)

print("\nGenerando augmentation combinada...")
all_new_rows = []

for label_str in ["mixture", "unproven"]:
    current = (train_df["label"] == label_str).sum()
    total_needed = TARGET - current
    if total_needed <= 0:
        print(f"\n[{label_str}] Ya tiene {current} ejemplos, no hace falta augmentar.")
        continue

    avail_random    = len(df_random[df_random["label"] == label_str])
    avail_backtrans = len(df_backtrans[df_backtrans["label"] == label_str])

    from_random    = min(avail_random, total_needed // 3)
    from_backtrans = min(avail_backtrans, total_needed // 3)
    from_synonyms  = total_needed - from_random - from_backtrans

    print(f"\n[{label_str}] Actuales: {current} | Target: {TARGET}")
    print(f"  +{from_random} random, +{from_backtrans} backtrans, +{from_synonyms} sinónimos")

    if from_random > 0:
        rows = df_random[df_random["label"] == label_str].sample(n=from_random, random_state=SEED).copy()
        all_new_rows.append(rows[["claim", "main_text", "label", "augmented", "aug_type"]])

    if from_backtrans > 0:
        rows = df_backtrans[df_backtrans["label"] == label_str].sample(n=from_backtrans, random_state=SEED).copy()
        all_new_rows.append(rows[["claim", "main_text", "label", "augmented", "aug_type"]])

    if from_synonyms > 0:
        subset = train_df[train_df["label"] == label_str].reset_index(drop=True)
        filas_nuevas = []
        textos_generados = set()
        while len(filas_nuevas) < from_synonyms:
            for _, row in subset.iterrows():
                if len(filas_nuevas) >= from_synonyms:
                    break
                nuevo_texto = augment_synonyms(str(row["main_text"]))
                if nuevo_texto not in textos_generados:
                    textos_generados.add(nuevo_texto)
                    nueva_fila = row.copy()
                    nueva_fila["main_text"] = nuevo_texto
                    nueva_fila["augmented"] = True
                    nueva_fila["aug_type"] = "synonyms"
                    filas_nuevas.append(nueva_fila)
        syn_df = pd.DataFrame(filas_nuevas)
        all_new_rows.append(syn_df[["claim", "main_text", "label", "augmented", "aug_type"]])

    print(f"  [{label_str}] completado")

train_df["aug_type"] = "original"
combined_df = pd.concat([train_df] + all_new_rows, ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("\n=== Distribución final ===")
print(combined_df["label"].value_counts())
print(f"Total filas: {len(combined_df)}")

output_path = DATA_PATH + "train_augmented_combined.csv"
combined_df.to_csv(output_path, index=False)
print(f"\nGuardado en: {output_path}")
