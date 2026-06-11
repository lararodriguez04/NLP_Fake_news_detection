# RoBERTa con estrategia head+tail para aprovechar mejor los 512 tokens
# En lugar de cortar por el final, coge el principio y el final del main_text

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding)
from sklearn.metrics import (classification_report, f1_score, accuracy_score)

MODEL_NAME = "roberta-base"
MODEL_SHORT = "roberta_headtail"
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 512

DATA_PATH = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/CODES/results/roberta_headtail"
os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"  Modelo: {MODEL_NAME}")
print(f"  Épocas: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max tokens: {MAX_LENGTH}")

df_train = pd.read_csv(DATA_PATH + "train_reduced.csv")
df_dev   = pd.read_csv(DATA_PATH + "dev_reduced.csv")
df_test  = pd.read_csv(DATA_PATH + "test_reduced.csv")

print(f"Train: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

for df, nombre in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
    antes = len(df)
    df.dropna(subset=["claim", "main_text"], inplace=True)
    if len(df) < antes:
        print(f"   {nombre}: eliminadas {antes - len(df)} filas con nulos")

LABEL2ID = {"true": 0, "false": 1, "mixture": 2, "unproven": 3}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

for df in [df_train, df_dev, df_test]:
    df["label"] = df["label"].map(LABEL2ID)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

print(f"\nDistribución de labels en train:")
for label, idx in LABEL2ID.items():
    count = (df_train["label"] == idx).sum()
    print(f"  {label:10}: {count:5} ({count/len(df_train)*100:.1f}%)")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_headtail(examples):
    result_input_ids = []
    result_attention = []

    for claim, main_text in zip(examples["claim"], examples["main_text"]):
        # Tokenizo el claim para saber cuántos tokens ocupa
        claim_tokens = tokenizer(claim, add_special_tokens=False)["input_ids"]

        # Los tokens disponibles para main_text son 512 menos claim, menos 3 tokens especiales (CLS, SEP, SEP)
        available = MAX_LENGTH - len(claim_tokens) - 3
        half = available // 2

        # Tokenizo el main_text completo sin truncar
        main_tokens = tokenizer(main_text, add_special_tokens=False)["input_ids"]

        # Cojo la primera mitad y la última mitad del main_text
        if len(main_tokens) <= available:
            selected = main_tokens
        else:
            selected = main_tokens[:half] + main_tokens[-half:]

        # Construyo la secuencia final: CLS + claim + SEP + main_text_headtail + SEP
        input_ids = [tokenizer.cls_token_id] + claim_tokens + [tokenizer.sep_token_id] + selected + [tokenizer.sep_token_id]
        attention_mask = [1] * len(input_ids)

        result_input_ids.append(input_ids)
        result_attention.append(attention_mask)

    return {"input_ids": result_input_ids, "attention_mask": result_attention}

train_dataset = Dataset.from_pandas(df_train[["claim", "main_text", "label"]].reset_index(drop=True))
dev_dataset   = Dataset.from_pandas(df_dev[["claim", "main_text", "label"]].reset_index(drop=True))
test_dataset  = Dataset.from_pandas(df_test[["claim", "main_text", "label"]].reset_index(drop=True))

print("\nTokenizando los datos")
train_dataset = train_dataset.map(tokenize_headtail, batched=True)
dev_dataset   = dev_dataset.map(tokenize_headtail, batched=True)
test_dataset  = test_dataset.map(tokenize_headtail, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
dev_dataset   = dev_dataset.rename_column("label", "labels")
test_dataset  = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch",   columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

print(f"\nCargando modelo {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS, id2label=ID2LABEL, label2id=LABEL2ID)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average="macro")
    accuracy  = accuracy_score(labels, predictions)
    return {"macro_f1": macro_f1, "accuracy": accuracy}

output_dir = os.path.join(RESULTS_PATH, MODEL_SHORT)
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
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

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("\nEmpezando train")
trainer.train()

print("\n Empezando test")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels_test = df_test["label"].values

print("\n RESULTADOS FINALES EN TEST")
print(classification_report(labels_test, preds, target_names=["true", "false", "mixture", "unproven"]))

macro_f1 = f1_score(labels_test, preds, average="macro")
print(f"Macro-F1 final: {macro_f1:.4f}")

probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

df_resultados = df_test.copy()
df_resultados["pred_label"]   = [ID2LABEL[p] for p in preds]
df_resultados["prob_true"]     = probs[:, 0]
df_resultados["prob_false"]    = probs[:, 1]
df_resultados["prob_mixture"]  = probs[:, 2]
df_resultados["prob_unproven"] = probs[:, 3]
df_resultados["max_prob"]      = probs.max(axis=-1)

results_file = os.path.join(RESULTS_PATH, f"predictions_{MODEL_SHORT}.csv")
df_resultados.to_csv(results_file, index=False)
print(f"\n Predicciones guardadas en: {results_file}")

trainer.save_model(os.path.join(output_dir, "best_model"))
print(f" Modelo guardado en: {output_dir}/best_model")
