# Longformer con tres tipos de augmentation combinados:
# random swap/deletion, back-translation EN-ES-EN y sinónimos BERT fill-mask
# mixture y unproven se igualan a 5078 ejemplos, false se deja sin augmentar

import os
import random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dataclasses import dataclass
from typing import Any
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    pipeline
)
from sklearn.metrics import classification_report, f1_score, accuracy_score

MODEL_NAME  = "allenai/longformer-base-4096"
MODEL_SHORT = "longformer_3aug"
EPOCHS      = 2
BATCH_SIZE  = 4
MAX_LENGTH  = 4096
AUGMENT_RATIO = 0.15
TARGET      = 5078
SEED        = 42
random.seed(SEED)

DATA_PATH    = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/CODES/results/longformer_3aug"
os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"  Modelo:     {MODEL_NAME}")
print(f"  Épocas:     {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max tokens: {MAX_LENGTH}")

df_train = pd.read_csv(DATA_PATH + "train_reduced.csv")
df_dev   = pd.read_csv(DATA_PATH + "dev_reduced.csv")
df_test  = pd.read_csv(DATA_PATH + "test_reduced.csv")

print(f"\nTrain: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

for df, nombre in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
    antes = len(df)
    df.dropna(subset=["claim", "main_text"], inplace=True)
    df = df[df["label"].isin(["true", "false", "mixture", "unproven"])]
    if len(df) < antes:
        print(f"  {nombre}: eliminadas {antes - len(df)} filas")

LABEL2ID = {"true": 0, "false": 1, "mixture": 2, "unproven": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

print(f"\nDistribución de labels en train antes de augmentation:")
for label in LABEL2ID:
    count = (df_train["label"] == label).sum()
    print(f"  {label:10}: {count:5} ({count/len(df_train)*100:.1f}%)")

df_random = pd.read_csv(DATA_PATH + "train_augmented_3.csv")
df_random = df_random[(df_random["augmented"] == True) & (df_random["label"].isin(LABEL2ID))][["claim", "main_text", "label"]].copy()

df_backtrans = pd.read_csv(DATA_PATH + "train_augmented_2.csv")
df_backtrans = df_backtrans[(df_backtrans["augmented"] == True) & (df_backtrans["label"].isin(LABEL2ID))].rename(columns={"text_1": "claim"})[["claim", "main_text", "label"]].copy()

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

def generate_synonyms(source_df, label_str, n_needed):
    subset = source_df[source_df["label"] == label_str].reset_index(drop=True)
    filas_nuevas = []
    textos_generados = set()
    while len(filas_nuevas) < n_needed:
        for _, row in subset.iterrows():
            if len(filas_nuevas) >= n_needed:
                break
            nuevo_texto = augment_synonyms(str(row["main_text"]))
            if nuevo_texto not in textos_generados:
                textos_generados.add(nuevo_texto)
                nueva_fila = row.copy()
                nueva_fila["main_text"] = nuevo_texto
                filas_nuevas.append(nueva_fila)
    return pd.DataFrame(filas_nuevas)

print("\nGenerando augmentation combinada...")
df_train_orig = df_train.copy()
all_new_rows = []

for label_str in ["mixture", "unproven"]:
    current = (df_train["label"] == label_str).sum()
    total_needed = TARGET - current
    if total_needed <= 0:
        continue

    avail_random    = len(df_random[df_random["label"] == label_str])
    avail_backtrans = len(df_backtrans[df_backtrans["label"] == label_str])

    from_random    = min(avail_random, total_needed // 3)
    from_backtrans = min(avail_backtrans, total_needed // 3)
    from_synonyms  = total_needed - from_random - from_backtrans

    if from_random > 0:
        rows = df_random[df_random["label"] == label_str].sample(n=from_random, random_state=SEED)
        all_new_rows.append(rows[["claim", "main_text", "label"]])

    if from_backtrans > 0:
        rows = df_backtrans[df_backtrans["label"] == label_str].sample(n=from_backtrans, random_state=SEED)
        all_new_rows.append(rows[["claim", "main_text", "label"]])

    if from_synonyms > 0:
        rows = generate_synonyms(df_train_orig, label_str, from_synonyms)
        all_new_rows.append(rows[["claim", "main_text", "label"]])

    print(f"  {label_str}: +{from_random} random, +{from_backtrans} backtrans, +{from_synonyms} sinónimos")

df_train = pd.concat([df_train] + all_new_rows, ignore_index=True)
df_train = df_train[df_train["label"].isin(LABEL2ID)].copy()

print(f"\nDistribución de labels en train después de augmentation:")
for label in LABEL2ID:
    count = (df_train["label"] == label).sum()
    print(f"  {label:10}: {count:5} ({count/len(df_train)*100:.1f}%)")

for df in [df_train, df_dev, df_test]:
    df["label"] = df["label"].map(LABEL2ID)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    encoding = tokenizer(
        examples["claim"],
        examples["main_text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )
    global_attention_masks = []
    for input_ids in encoding["input_ids"]:
        mask = [0] * len(input_ids)
        mask[0] = 1
        global_attention_masks.append(mask)
    encoding["global_attention_mask"] = global_attention_masks
    return encoding

train_dataset = Dataset.from_pandas(df_train[["claim", "main_text", "label"]].reset_index(drop=True))
dev_dataset   = Dataset.from_pandas(df_dev[["claim", "main_text", "label"]].reset_index(drop=True))
test_dataset  = Dataset.from_pandas(df_test[["claim", "main_text", "label"]].reset_index(drop=True))

print("\nTokenizando los datos...")
train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset   = dev_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
dev_dataset   = dev_dataset.rename_column("label", "labels")
test_dataset  = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])
dev_dataset.set_format("torch",   columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])
test_dataset.set_format("torch",  columns=["input_ids", "attention_mask", "global_attention_mask", "labels"])

print(f"\nCargando modelo {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    use_safetensors=True
)

# class weights actualizados con la nueva distribución tras augmentation
class_counts = torch.tensor([5078, 3001, 5078, 5078], dtype=torch.float)
class_weights = class_counts.sum() / (len(class_counts) * class_counts)
class_weights = class_weights.to("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average="macro")
    accuracy  = accuracy_score(labels, predictions)
    return {"macro_f1": macro_f1, "accuracy": accuracy}

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False, **kwargs):
        labels = inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=class_weights)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

output_dir = os.path.join(RESULTS_PATH, MODEL_SHORT)
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=50,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    report_to="none"
)

@dataclass
class LongformerDataCollator(DataCollatorWithPadding):
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        global_attention_masks = [f.pop("global_attention_mask") for f in features]
        batch = super().__call__(features)
        max_len = batch["input_ids"].shape[1]
        padded = []
        for mask in global_attention_masks:
            if isinstance(mask, torch.Tensor):
                mask = mask.tolist()
            padding_length = max_len - len(mask)
            padded.append(mask + [0] * padding_length)
        batch["global_attention_mask"] = torch.tensor(padded, dtype=torch.long)
        return batch

data_collator = LongformerDataCollator(tokenizer=tokenizer)

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("\nEmpezando train...")
trainer.train()

print("\nEmpezando test...")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels_test = df_test["label"].values

print("\nRESULTADOS FINALES EN TEST")
print(classification_report(labels_test, preds, target_names=["true", "false", "mixture", "unproven"]))
macro_f1 = f1_score(labels_test, preds, average="macro")
print(f"Macro-F1 final: {macro_f1:.4f}")

probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()
df_resultados = df_test.copy()
df_resultados["pred_label"]    = [ID2LABEL[p] for p in preds]
df_resultados["prob_true"]     = probs[:, 0]
df_resultados["prob_false"]    = probs[:, 1]
df_resultados["prob_mixture"]  = probs[:, 2]
df_resultados["prob_unproven"] = probs[:, 3]
df_resultados["max_prob"]      = probs.max(axis=-1)

results_file = os.path.join(RESULTS_PATH, f"predictions_{MODEL_SHORT}.csv")
df_resultados.to_csv(results_file, index=False)
print(f"\nPredicciones guardadas en: {results_file}")
trainer.save_model(os.path.join(output_dir, "best_model"))
print(f"Modelo guardado en: {output_dir}/best_model")
