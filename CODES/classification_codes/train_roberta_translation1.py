# Entreno de lasificador de fake news sobre el dataset PubHealth usando RoBERTa
# Input: claim + main_text → Output: true / false / mixture / unproven

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
from sklearn.metrics import (classification_report,f1_score,accuracy_score)


#CONFIGURACIÓN
MODEL_NAME = "roberta-base"
MODEL_SHORT = "roberta"
EPOCHS = 3
BATCH_SIZE = 16
MAX_LENGTH = 512

DATA_PATH = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/CODES/results/roberta"
os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"  Modelo: {MODEL_NAME}")
print(f"  Épocas: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max tokens: {MAX_LENGTH}")

#CARGAR LOS DATOS DESDE LOS CSV LOCALES
df_train = pd.read_csv(DATA_PATH + "train_augmented_2.csv")
df_dev = pd.read_csv(DATA_PATH + "dev_reduced.csv")
df_test  = pd.read_csv(DATA_PATH + "test_reduced.csv")

print(f"Train: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

# BEFORE loading into datasets, rename dev and test
df_dev = df_dev.rename(columns={"claim": "text_1", "explanation": "text_2"})
df_test = df_test.rename(columns={"claim": "text_1", "explanation": "text_2"})

# Then the loop only needs to drop nulls
for df, nombre in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
    antes = len(df)
    df.dropna(subset=["text_1", "main_text"], inplace=True)
    if len(df) < antes:
        print(f"   {nombre}: eliminadas {antes - len(df)} filas con nulos")

#MAPEO DE LABELS A NÚMEROS
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


#TOKENIZACIÓN
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    # Concatenar text_1 + main_text como par de textos separados por [SEP]
    return tokenizer(
        examples["text_1"],
        examples["main_text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False)

# Convierto los DataFrames a datasets de HuggingFace
train_dataset = Dataset.from_pandas(df_train[["text_1", "main_text", "label"]].reset_index(drop=True))
dev_dataset = Dataset.from_pandas(df_dev[["text_1", "main_text", "label"]].reset_index(drop=True))
test_dataset = Dataset.from_pandas(df_test[["text_1", "main_text", "label"]].reset_index(drop=True))

print("\nTokenizando los datos")
train_dataset = train_dataset.map(tokenize, batched=True)
dev_dataset = dev_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
dev_dataset = dev_dataset.rename_column("label", "labels")
test_dataset = test_dataset.rename_column("label", "labels")

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])


#MODELO
print(f"\nCargando modelo {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME,num_labels=NUM_LABELS,id2label=ID2LABEL,label2id=LABEL2ID)


#MÉTRICAS --> Macro-F1 como métrica principal

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    macro_f1 = f1_score(labels, predictions, average="macro")
    accuracy  = accuracy_score(labels, predictions)
    return {"macro_f1": macro_f1,"accuracy": accuracy}


#TRAINING ARGUMENTS
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


#TRAINER

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


#ENTRENAMIENTO

print("\nEmpezando train")
trainer.train()

#EVALUACIÓN FINAL EN TEST

print("\n Empezando test")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels_test = df_test["label"].values

print("\n RESULTADOS FINALES EN TEST")
print(classification_report(labels_test,preds,target_names=["true", "false", "mixture", "unproven"]))

macro_f1 = f1_score(labels_test, preds, average="macro")
print(f"Macro-F1 final: {macro_f1:.4f}")


# GUARDAR RESULTADOS CON PROBABILIDADES (para el threshold)

probs = torch.softmax(torch.tensor(predictions.predictions), dim=-1).numpy()

df_resultados = df_test.copy()
df_resultados["pred_label"] = [ID2LABEL[p] for p in preds]
df_resultados["prob_true"] = probs[:, 0]
df_resultados["prob_false"]  = probs[:, 1]
df_resultados["prob_mixture"]  = probs[:, 2]
df_resultados["prob_unproven"] = probs[:, 3]
df_resultados["max_prob"] = probs.max(axis=-1)

results_file = os.path.join(RESULTS_PATH, f"predictions_{MODEL_SHORT}.csv")
df_resultados.to_csv(results_file, index=False)
print(f"\n Predicciones guardadas en: {results_file}")

trainer.save_model(os.path.join(output_dir, "best_model"))
print(f" Modelo guardado en: {output_dir}/best_model")
