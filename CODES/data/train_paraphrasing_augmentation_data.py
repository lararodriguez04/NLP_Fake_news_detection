from transformers import BartForConditionalGeneration, BartTokenizer
import pandas as pd
import torch

DATA_PATH = "/home/lnlpG08/nlp/DATA/"
BART_PATH = "facebook/bart-large"  # Puedes usar un checkpoint local si lo has descargado previamente

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# LOAD MODEL
print("Loading BART...")
tok_bart   = BartTokenizer.from_pretrained(BART_PATH)
model_bart = BartForConditionalGeneration.from_pretrained(BART_PATH).to(device)
print("BART loaded.")


def paraphrase(texts, batch_size=8):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tok_bart(
            batch, return_tensors="pt", padding=True,
            truncation=True, max_length=512
        ).to(device)
        with torch.no_grad():
            outputs = model_bart.generate(
                **inputs,
                num_beams=4,
                max_length=512,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
        decoded = tok_bart.batch_decode(outputs, skip_special_tokens=True)
        results.extend(decoded)
        print(f"  {min(i+batch_size, len(texts))}/{len(texts)}")
    return results


# LOAD DATA
print("\nLoading data...")
df_train = pd.read_csv(DATA_PATH + "train_reduced.csv")
df_train.dropna(subset=["claim", "main_text"], inplace=True)

df_unproven = df_train[df_train["label"] == "unproven"].copy()
df_mixture  = df_train[df_train["label"] == "mixture"].copy()

print(f"unproven: {len(df_unproven)} | mixture: {len(df_mixture)}")

# AUGMENT UNPROVEN x2 (291 → ~873)
print("\nAugmenting unproven [1/2]...")
aug_unproven_1 = df_unproven.copy()
aug_unproven_1["claim"]     = paraphrase(df_unproven["claim"].tolist())
aug_unproven_1["main_text"] = paraphrase(df_unproven["main_text"].tolist())

print("\nAugmenting unproven [2/2]...")
aug_unproven_2 = df_unproven.copy()
aug_unproven_2["claim"]     = paraphrase(df_unproven["claim"].tolist())
aug_unproven_2["main_text"] = paraphrase(df_unproven["main_text"].tolist())

# AUGMENT MIXTURE x1 (1434 → ~2868)
print("\nAugmenting mixture [1/1]...")
aug_mixture = df_mixture.copy()
aug_mixture["claim"]     = paraphrase(df_mixture["claim"].tolist())
aug_mixture["main_text"] = paraphrase(df_mixture["main_text"].tolist())

# COMBINE AND SHUFFLE
print("\nCombining datasets...")
df_train_aug = pd.concat(
    [df_train, aug_unproven_1, aug_unproven_2, aug_mixture],
    ignore_index=True
).sample(frac=1, random_state=42)

df_train_aug.to_csv(DATA_PATH + "train_augmented.csv", index=False)

print(f"\nDone!")
print(f"Original : {len(df_train)}")
print(f"Augmented: {len(df_train_aug)}")
print(f"\nLabel distribution:")
print(df_train_aug["label"].value_counts())