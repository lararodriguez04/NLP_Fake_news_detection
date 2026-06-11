
#   true     : ~5078  
#   false    : ~3001  
#   mixture  : ~3000  
#   unproven : ~1200 

import random
import pandas as pd

SEED = 42
random.seed(SEED)

DATA_PATH = "/home/lnlpG08/nlp/DATA/"

AUGMENT_TARGETS = {
    "mixture":  3000,
    "unproven": 1200,
}

# Parámetros de augmentation
P_DELETE = 0.1    # probabilidad de eliminar cada palabra (10%)
N_SWAP   = 3      # número de intercambios aleatorios por texto


# ── Funciones de augmentation ─────────────────────────────────────────────────

def random_deletion(text, p=P_DELETE):
    
    words = text.split()
    # Si el texto es muy corto, no eliminar nada
    if len(words) <= 5:
        return text
    new_words = [w for w in words if random.random() > p]
    # Si se eliminaron todas, devolver una palabras aleatorias
    if len(new_words) == 0:
        return random.choice(words)
    return " ".join(new_words)


def random_swap(text, n=N_SWAP):
    
    words = text.split()
    # Si el texto es muy corto, no intercambiar nada
    if len(words) <= 3:
        return text
    new_words = words.copy()
    for _ in range(n):
        idx1, idx2 = random.sample(range(len(new_words)), 2)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return " ".join(new_words)


def augment_text(text):
    
    text = random_swap(str(text))
    text = random_deletion(text)
    return text


# ── Main ──────────────────────────────────────────────────────────────────────

print("Cargando datos...")
train_df = pd.read_csv(DATA_PATH + "train_reduced.csv")
train_df.dropna(subset=["claim", "main_text"], inplace=True)

print(f"Train size: {len(train_df)}")
print("\nDistribución original:")
print(train_df["label"].value_counts())

augmented_rows = []

for label, target in AUGMENT_TARGETS.items():
    subset = train_df[train_df["label"] == label].reset_index(drop=True)
    current = len(subset)
    need = target - current

    if need <= 0:
        print(f"\n[{label}] Ya tiene {current} ejemplos, no hace falta augmentar.")
        continue

    print(f"\n[{label}] Actuales: {current} | Necesarios: {need} | Target: {target}")

    # Muestrear con reemplazo si necesitamos más de los que hay
    sample_pool = subset.sample(n=need, replace=(need > current), random_state=SEED)

    for _, row in sample_pool.iterrows():
        nueva_fila = row.copy()
        # Solo augmentamos el main_text, el claim lo dejamos igual
        nueva_fila["main_text"] = augment_text(str(row["main_text"]))
        nueva_fila["augmented"] = True
        augmented_rows.append(nueva_fila)

    print(f"  Generados {len(augmented_rows)} ejemplos hasta ahora")

# Combinar y mezclar
aug_df = pd.DataFrame(augmented_rows)
train_df["augmented"] = False

combined_df = pd.concat([train_df, aug_df], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

print("\n=== Distribución final (augmented train) ===")
print(combined_df["label"].value_counts())
print(f"Total filas: {len(combined_df)}")

output_path = DATA_PATH + "train_augmented_3.csv"
combined_df.to_csv(output_path, index=False)
print(f"\nGuardado en: {output_path}")
