
# Fine-tuning BART-Large para generación de explicaciones en PubHealth
# MODIFICADO: guarda también las explicaciones de train y dev (no solo test)
# MODIFICADO: save_total_limit=1 para no llenar el disco con checkpoints
 
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
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
MODEL_NAME  = "facebook/bart-large"
MODEL_SHORT = "bart-explanation-genexp"
EPOCHS      = 5
BATCH_SIZE  = 8
MAX_INPUT   = 512
MAX_TARGET  = 128
 
DATA_PATH    = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/RESULTS/bart_explanation_genexp/"
os.makedirs(RESULTS_PATH, exist_ok=True)
 
print(f"  Modelo: {MODEL_NAME}")
print(f"  Épocas: {EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max input tokens: {MAX_INPUT}")
print(f"  Max target tokens: {MAX_TARGET}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
df_train = pd.read_csv(DATA_PATH + "train_reduced.csv")
df_dev   = pd.read_csv(DATA_PATH + "dev_reduced.csv")
df_test  = pd.read_csv(DATA_PATH + "test_reduced.csv")
 
print(f"\nTrain: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")
 
for df, nombre in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
    antes = len(df)
    df.dropna(subset=["claim", "main_text", "explanation"], inplace=True)
    if len(df) < antes:
        print(f"  {nombre}: eliminadas {antes - len(df)} filas con nulos")
 
print(f"\nTrain: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. TOKENIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
 
def tokenize(examples):
    inputs = [
        f"claim: {c} article: {a}"
        for c, a in zip(examples["claim"], examples["main_text"])
    ]
    targets = examples["explanation"]
 
    model_inputs = tokenizer(
        inputs, max_length=MAX_INPUT, truncation=True, padding=False
    )
    labels = tokenizer(
        text_target=targets, max_length=MAX_TARGET, truncation=True, padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
 
train_dataset = Dataset.from_pandas(df_train[["claim", "main_text", "explanation"]].reset_index(drop=True))
dev_dataset   = Dataset.from_pandas(df_dev[["claim", "main_text", "explanation"]].reset_index(drop=True))
test_dataset  = Dataset.from_pandas(df_test[["claim", "main_text", "explanation"]].reset_index(drop=True))
 
# También necesitamos datasets sin tokenizar para la inferencia final
train_dataset_raw = Dataset.from_pandas(df_train[["claim", "main_text", "explanation"]].reset_index(drop=True))
dev_dataset_raw   = Dataset.from_pandas(df_dev[["claim", "main_text", "explanation"]].reset_index(drop=True))
 
print("\nTokenizando los datos...")
train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=train_dataset.column_names)
dev_dataset   = dev_dataset.map(tokenize, batched=True, remove_columns=dev_dataset.column_names)
test_dataset  = test_dataset.map(tokenize, batched=True, remove_columns=test_dataset.column_names)
 
train_dataset_tok = train_dataset_raw.map(tokenize, batched=True, remove_columns=train_dataset_raw.column_names)
dev_dataset_tok   = dev_dataset_raw.map(tokenize, batched=True, remove_columns=dev_dataset_raw.column_names)
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELO
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nCargando modelo {MODEL_NAME}...")
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────
rouge = evaluate.load("rouge")
bleu  = evaluate.load("bleu")
 
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.clip(predictions, 0, tokenizer.vocab_size - 1).astype(np.int32)
    labels      = np.where(labels != -100, labels, tokenizer.pad_token_id).astype(np.int32)
 
    decoded_preds  = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
    decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
 
    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bleu_result  = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
 
    return {
        "rouge1": round(rouge_result["rouge1"], 4),
        "rouge2": round(rouge_result["rouge2"], 4),
        "rougeL": round(rouge_result["rougeL"], 4),
        "bleu":   round(bleu_result["bleu"], 4),
    }
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING ARGUMENTS
# save_total_limit=1 → solo guarda el mejor checkpoint, borra los demás
# ─────────────────────────────────────────────────────────────────────────────
output_dir = os.path.join(RESULTS_PATH, MODEL_SHORT)
 
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,              # ← SOLO GUARDA EL MEJOR CHECKPOINT
    load_best_model_at_end=True,
    metric_for_best_model="rougeL",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET,
    logging_dir=os.path.join(output_dir, "logs"),
    logging_steps=50,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    report_to="none"
)
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. TRAINER
# ─────────────────────────────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
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
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
print("\nEmpezando train...")
trainer.train()
 
# ─────────────────────────────────────────────────────────────────────────────
# 8. INFERENCIA EN TEST
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerando explicaciones para TEST...")
test_results   = trainer.predict(test_dataset)
predictions    = np.clip(test_results.predictions, 0, tokenizer.vocab_size - 1).astype(np.int32)
labels         = np.where(test_results.label_ids != -100, test_results.label_ids, tokenizer.pad_token_id).astype(np.int32)
decoded_preds  = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
decoded_labels = [l.strip() for l in tokenizer.batch_decode(labels, skip_special_tokens=True)]
 
rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
bleu_result  = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
 
print("\nRESULTADOS FINALES EN TEST")
print(f"  ROUGE-1: {rouge_result['rouge1']:.4f}")
print(f"  ROUGE-2: {rouge_result['rouge2']:.4f}")
print(f"  ROUGE-L: {rouge_result['rougeL']:.4f}")
print(f"  BLEU:    {bleu_result['bleu']:.4f}")
 
df_test_out = df_test.copy()
df_test_out["generated_explanation"]  = decoded_preds
df_test_out["reference_explanation"]  = decoded_labels
df_test_out.to_csv(os.path.join(RESULTS_PATH, "test_bart_genexp.csv"), index=False)
print(f"\nTest guardado en: {RESULTS_PATH}test_bart_genexp.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# 9. INFERENCIA EN TRAIN
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerando explicaciones para TRAIN...")
train_results  = trainer.predict(train_dataset_tok)
predictions    = np.clip(train_results.predictions, 0, tokenizer.vocab_size - 1).astype(np.int32)
decoded_preds  = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
 
df_train_out = df_train.copy()
df_train_out["generated_explanation"] = decoded_preds
df_train_out.to_csv(os.path.join(RESULTS_PATH, "train_bart_genexp.csv"), index=False)
print(f"Train guardado en: {RESULTS_PATH}train_bart_genexp.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# 10. INFERENCIA EN DEV
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerando explicaciones para DEV...")
dev_results   = trainer.predict(dev_dataset_tok)
predictions   = np.clip(dev_results.predictions, 0, tokenizer.vocab_size - 1).astype(np.int32)
decoded_preds = [p.strip() for p in tokenizer.batch_decode(predictions, skip_special_tokens=True)]
 
df_dev_out = df_dev.copy()
df_dev_out["generated_explanation"] = decoded_preds
df_dev_out.to_csv(os.path.join(RESULTS_PATH, "dev_bart_genexp.csv"), index=False)
print(f"Dev guardado en: {RESULTS_PATH}dev_bart_genexp.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# 11. GUARDAR MODELO
# ─────────────────────────────────────────────────────────────────────────────
trainer.save_model(os.path.join(output_dir, "best_model"))
print(f"\nModelo guardado en: {output_dir}/best_model")
print("\n¡Todo listo!")