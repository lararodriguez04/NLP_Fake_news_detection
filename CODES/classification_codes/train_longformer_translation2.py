

import os
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
    DataCollatorWithPadding
)
from sklearn.metrics import classification_report, f1_score, accuracy_score




MODEL_NAME  = "allenai/longformer-base-4096"
MODEL_SHORT = "longformer"
EPOCHS      = 2
BATCH_SIZE  = 4     
MAX_LENGTH  = 4096  

DATA_PATH    = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/CODES/results/"
os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"  Modelo:     {MODEL_NAME}")
print(f"  Épocas:     {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max tokens: {MAX_LENGTH}")


# CARGAR DATOS DESDE LOS CSV LOCALES

df_train = pd.read_csv(DATA_PATH + "train_augmented_data_train.csv")
df_dev   = pd.read_csv(DATA_PATH + "dev_reduced.csv")
df_test  = pd.read_csv(DATA_PATH + "test_reduced.csv")

print(f"\nTrain: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

# BEFORE loading into datasets, rename dev and test
df_dev = df_dev.rename(columns={"claim": "text_1", "explanation": "text_2"})
df_test = df_test.rename(columns={"claim": "text_1", "explanation": "text_2"})

# Then the loop only needs to drop nulls
for df, nombre in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
    antes = len(df)
    df.dropna(subset=["text_1", "main_text"], inplace=True)
    if len(df) < antes:
        print(f"   {nombre}: eliminadas {antes - len(df)} filas con nulos")
        
# MAPEO DE LABELS A NÚMEROS

LABEL2ID   = {"true": 0, "false": 1, "mixture": 2, "unproven": 3}
ID2LABEL   = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS = len(LABEL2ID)

for df in [df_train, df_dev, df_test]:
    df["label"] = df["label"].map(LABEL2ID)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

print(f"\nDistribución de labels en train:")
for label, idx in LABEL2ID.items():
    count = (df_train["label"] == idx).sum()
    print(f"  {label:10}: {count:5} ({count/len(df_train)*100:.1f}%)")


# TOKENIZACIÓN

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):

    encoding = tokenizer(
        examples["text_1"],
        examples["main_text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False        
    )

    # Crear global_attention_mask: 1 solo en [CLS] (posición 0), 0 en el resto
    global_attention_masks = []
    for input_ids in encoding["input_ids"]:
        mask = [0] * len(input_ids)
        mask[0] = 1   
        global_attention_masks.append(mask)

    encoding["global_attention_mask"] = global_attention_masks
    return encoding


# Convertir DataFrames a datasets de HuggingFace
train_dataset = Dataset.from_pandas(df_train[["text_1", "main_text", "label"]].reset_index(drop=True))
dev_dataset   = Dataset.from_pandas(df_dev[["text_1", "main_text", "label"]].reset_index(drop=True))
test_dataset  = Dataset.from_pandas(df_test[["text_1", "main_text", "label"]].reset_index(drop=True))

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


# MODELO

print(f"\nCargando modelo {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    use_safetensors=True
)

# MÉTRICAS 


class_counts = torch.tensor([5078, 3001, 1434, 291], dtype=torch.float)
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
            loss = torch.nn.functional.cross_entropy(
                logits, labels, weight=class_weights
            )
        else:
            loss = outputs.loss

        return (loss, outputs) if return_outputs else loss
        
# TRAINING ARGUMENTS

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


# TRAINER

@dataclass
class LongformerDataCollator(DataCollatorWithPadding):
  
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        # Extraemos la global_attention_mask antes de que el collator base la procese
        global_attention_masks = [f.pop("global_attention_mask") for f in features]

        # El collator base procesa input_ids, attention_mask y labels
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


# TRAINING

print("\nEmpezando train...")
trainer.train()


# EVALUACIÓN FINAL EN TEST

print("\nEmpezando test...")
predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels_test = df_test["label"].values

print("\nRESULTADOS FINALES EN TEST")
print(classification_report(labels_test, preds, target_names=["true", "false", "mixture", "unproven"]))

macro_f1 = f1_score(labels_test, preds, average="macro")
print(f"Macro-F1 final: {macro_f1:.4f}")


# GUARDAR RESULTADOS CON PROBABILIDADES (para el threshold)

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
