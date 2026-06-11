# Fine-tuning BART for explanation generation on PubHealth
# Input: text_1 (claim) + main_text → Output: explanation

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate

# 0. CONFIGURACIÓN

MODEL_NAME  = "facebook/bart-large"
MODEL_SHORT = "bart-explanation"
EPOCHS      = 5
BATCH_SIZE  = 8
MAX_INPUT   = 512
MAX_TARGET  = 128

DATA_PATH    = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/CODES/results/bart_explanation/"
os.makedirs(RESULTS_PATH, exist_ok=True)

print(f"  Modelo: {MODEL_SHORT}")
print(f"  Épocas: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max input tokens: {MAX_INPUT}")
print(f"  Max target tokens: {MAX_TARGET}")

# 1. CARGAR DATOS

df_train = pd.read_csv(DATA_PATH + "train_reduced.csv")
df_dev   = pd.read_csv(DATA_PATH + "dev_reduced.csv")
df_test  = pd.read_csv(DATA_PATH + "test_reduced.csv")

print(f"\nTrain: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

# Eliminar nulos en las columnas relevantes
for df, nombre in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
    antes = len(df)
    df.dropna(subset=["claim", "main_text", "explanation"], inplace=True)
    if len(df) < antes:
        print(f"  {nombre}: eliminadas {antes - len(df)} filas con nulos")

print(f"\nTrain: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

# 2. TOKENIZACIÓN

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    # Input: claim + main_text concatenated with separator
    inputs = [
        f"claim: {c} article: {a}"
        for c, a in zip(examples["claim"], examples["main_text"])
    ]
    # Target: explanation (text_2)
    targets = examples["explanation"]

    model_inputs = tokenizer(
        inputs,
        max_length=MAX_INPUT,
        truncation=True,
        padding=False
    )
    labels = tokenizer(
        text_target=targets,
        max_length=MAX_TARGET,
        truncation=True,
        padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = Dataset.from_pandas(df_train[["claim", "main_text", "explanation"]].reset_index(drop=True))
dev_dataset   = Dataset.from_pandas(df_dev[["claim", "main_text", "explanation"]].reset_index(drop=True))
test_dataset  = Dataset.from_pandas(df_test[["claim", "main_text", "explanation"]].reset_index(drop=True))

print("\nTokenizando los datos...")
train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
dev_dataset   = dev_dataset.map(tokenize, batched=True, remove_columns=dev_dataset.column_names)
test_dataset  = test_dataset.map(tokenize, batched=True, remove_columns=test_dataset.column_names)

# 3. MODELO

print(f"\nCargando modelo...")
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

# 4. MÉTRICAS — ROUGE + BLEU

rouge = evaluate.load("rouge")
bleu  = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1).astype(np.int32)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = labels.astype(np.int32)

    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip whitespace
    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    rouge_result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True
    )

    bleu_result = bleu.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels]
    )

    return {
        "rouge1":  round(rouge_result["rouge1"], 4),
        "rouge2":  round(rouge_result["rouge2"], 4),
        "rougeL":  round(rouge_result["rougeL"], 4),
        "bleu":    round(bleu_result["bleu"], 4),
    }

# 5. TRAINING ARGUMENTS

output_dir = os.path.join(RESULTS_PATH, MODEL_SHORT)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    predict_with_generate=True,          # needed for seq2seq metrics
    generation_max_length=MAX_TARGET,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=50,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    report_to="none"
)

# 6. TRAINER

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    label_pad_token_id=-100
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 7. ENTRENAMIENTO

print("\nEmpezando train...")
trainer.train()

# 8. TEST — generate explanations and compute metrics

print("\nEmpezando test...")
test_results = trainer.predict(test_dataset)

# Decode predictions
predictions = test_results.predictions
labels      = test_results.label_ids
labels      = np.where(labels != -100, labels, tokenizer.pad_token_id)

predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1).astype(np.int32)
labels = labels.astype(np.int32)
decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

decoded_preds  = [p.strip() for p in decoded_preds]
decoded_labels = [l.strip() for l in decoded_labels]

# Compute final metrics
rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
bleu_result  = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

print("\nRESULTADOS FINALES EN TEST")
print(f"  ROUGE-1: {rouge_result['rouge1']:.4f}")
print(f"  ROUGE-2: {rouge_result['rouge2']:.4f}")
print(f"  ROUGE-L: {rouge_result['rougeL']:.4f}")
print(f"  BLEU:    {bleu_result['bleu']:.4f}")

# 9. GUARDAR PREDICCIONES

df_resultados = df_test.copy()
df_resultados["generated_explanation"] = decoded_preds
df_resultados["reference_explanation"] = decoded_labels  # same, but df_test now has "explanation" column already

results_file = os.path.join(RESULTS_PATH, "predictions_bart_explanation.csv")
df_resultados.to_csv(results_file, index=False)
print(f"\nPredicciones guardadas en: {results_file}")

trainer.save_model(os.path.join(output_dir, "best_model"))
print(f"Modelo guardado en: {output_dir}/best_model")