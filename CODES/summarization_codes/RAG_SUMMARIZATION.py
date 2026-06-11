# RAG-based explanation generation for PubHealth
# Pipeline: chunk main_text → retrieve relevant chunks with BERT → generate explanation with BART

import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModel,
    BartForConditionalGeneration, BartTokenizer,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from tqdm import tqdm

# 0. CONFIGURACIÓN

BERT_PATH = "bert-base-uncased"  # Usamos un BERT preentrenado estándar para el retriever
BART_PATH = "facebook/bart-base"  # Usamos un BART preentrenado estándar para el generador

MODEL_SHORT  = "rag-bart-explanation"
EPOCHS       = 5
BATCH_SIZE   = 8
MAX_INPUT    = 512
MAX_TARGET   = 128
CHUNK_SIZE   = 100    # tokens per chunk
CHUNK_OVERLAP = 20   # overlap between chunks
TOP_K        = 3     # number of chunks to retrieve

DATA_PATH    = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/RESULTS/rag_bart_explanation/"
os.makedirs(RESULTS_PATH, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Chunk size: {CHUNK_SIZE} | Overlap: {CHUNK_OVERLAP} | Top-K: {TOP_K}")

# 1. CARGAR DATOS

df_train = pd.read_csv(DATA_PATH + "train_reduced.csv")
df_dev   = pd.read_csv(DATA_PATH + "dev_reduced.csv")
df_test  = pd.read_csv(DATA_PATH + "test_reduced.csv")

for df, nombre in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
    antes = len(df)
    df.dropna(subset=["claim", "main_text", "explanation"], inplace=True)
    if len(df) < antes:
        print(f"  {nombre}: eliminadas {antes - len(df)} filas con nulos")

print(f"Train: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

# 2. BERT RETRIEVER

print("\nLoading BERT retriever...")
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert_model     = AutoModel.from_pretrained(BERT_PATH).to(device)
bert_model.eval()


def get_cls_embedding(text, max_length=128):
    """Get CLS embedding for a text using BERT."""
    inputs = bert_tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=max_length,
        padding=True
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()


def chunk_text(text, tokenizer, chunk_size=100, overlap=20):
    """Split text into overlapping token-based chunks."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text_decoded = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text_decoded)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def retrieve_top_chunks(claim, main_text, k=3):
    """Retrieve top-k most relevant chunks from main_text given a claim."""
    chunks = chunk_text(main_text, bert_tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)

    if len(chunks) == 0:
        return main_text[:500]

    if len(chunks) <= k:
        return " ".join(chunks)

    # Embed claim
    claim_emb = get_cls_embedding(claim)

    # Embed all chunks
    chunk_embs = np.array([get_cls_embedding(c) for c in chunks])

    # Cosine similarity
    claim_norm  = claim_emb / (np.linalg.norm(claim_emb) + 1e-8)
    chunks_norm = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    scores      = chunks_norm @ claim_norm

    # Get top-k indices in order of appearance (not score) for coherence
    top_indices = sorted(np.argsort(scores)[-k:].tolist())
    return " ".join([chunks[i] for i in top_indices])


# 3. BUILD RAG INPUTS

print("\nBuilding RAG inputs (retrieving relevant chunks)...")
print("This may take a few minutes...")


def build_rag_inputs(df, split_name):
    rag_inputs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
        retrieved = retrieve_top_chunks(
            row["claim"], row["main_text"], k=TOP_K
        )
        rag_input = f"claim: {row['claim']} evidence: {retrieved}"
        rag_inputs.append(rag_input)
    return rag_inputs


train_rag_inputs = build_rag_inputs(df_train, "train")
dev_rag_inputs   = build_rag_inputs(df_dev,   "dev")
test_rag_inputs  = build_rag_inputs(df_test,  "test")

# Save retrieved inputs for inspection
df_train_rag = df_train.copy()
df_train_rag["rag_input"] = train_rag_inputs
df_train_rag.to_csv(DATA_PATH + "train_rag_inputs.csv", index=False)
print("RAG inputs saved.")

# 4. TOKENIZE FOR BART

print("\nLoading BART generator...")
bart_tokenizer = BartTokenizer.from_pretrained(BART_PATH)
bart_model     = BartForConditionalGeneration.from_pretrained(BART_PATH)


def tokenize_rag(inputs, targets):
    model_inputs = bart_tokenizer(
        inputs,
        max_length=MAX_INPUT,
        truncation=True,
        padding=False
    )
    labels = bart_tokenizer(
        text_target=targets,
        max_length=MAX_TARGET,
        truncation=True,
        padding=False
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_targets = df_train["explanation"].tolist()
dev_targets   = df_dev["explanation"].tolist()
test_targets  = df_test["explanation"].tolist()

# Build HuggingFace datasets
train_dataset = Dataset.from_dict({
    "input":  train_rag_inputs,
    "target": train_targets
})
dev_dataset = Dataset.from_dict({
    "input":  dev_rag_inputs,
    "target": dev_targets
})
test_dataset = Dataset.from_dict({
    "input":  test_rag_inputs,
    "target": test_targets
})

print("\nTokenizing...")
train_dataset = train_dataset.map(
    lambda x: tokenize_rag(x["input"], x["target"]),
    batched=True, remove_columns=["input", "target"]
)
dev_dataset = dev_dataset.map(
    lambda x: tokenize_rag(x["input"], x["target"]),
    batched=True, remove_columns=["input", "target"]
)
test_dataset = test_dataset.map(
    lambda x: tokenize_rag(x["input"], x["target"]),
    batched=True, remove_columns=["input", "target"]
)

# 5. MÉTRICAS

rouge = evaluate.load("rouge")
bleu  = evaluate.load("bleu")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.clip(predictions, 0, bart_tokenizer.vocab_size - 1).astype(np.int32)
    labels = np.where(labels != -100, labels, bart_tokenizer.pad_token_id).astype(np.int32)

    decoded_preds  = [p.strip() for p in bart_tokenizer.batch_decode(predictions, skip_special_tokens=True)]
    decoded_labels = [l.strip() for l in bart_tokenizer.batch_decode(labels, skip_special_tokens=True)]

    rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    bleu_result  = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

    return {
        "rouge1": round(rouge_result["rouge1"], 4),
        "rouge2": round(rouge_result["rouge2"], 4),
        "rougeL": round(rouge_result["rougeL"], 4),
        "bleu":   round(bleu_result["bleu"], 4),
    }


# 6. TRAINING ARGUMENTS

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
    predict_with_generate=True,
    generation_max_length=MAX_TARGET,
    logging_steps=50,
    warmup_ratio=0.1,
    weight_decay=0.01,
    fp16=True,
    max_grad_norm=1.0,
    report_to="none"
)

# 7. TRAINER

data_collator = DataCollatorForSeq2Seq(
    tokenizer=bart_tokenizer,
    model=bart_model,
    padding=True,
    label_pad_token_id=-100
)

trainer = Seq2SeqTrainer(
    model=bart_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    processing_class=bart_tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 8. TRAIN

print("\nEmpezando train...")
trainer.train()

# 9. TEST

print("\nEmpezando test...")
test_results = trainer.predict(test_dataset)
predictions  = np.clip(test_results.predictions, 0, bart_tokenizer.vocab_size - 1).astype(np.int32)
labels       = np.where(test_results.label_ids != -100, test_results.label_ids, bart_tokenizer.pad_token_id).astype(np.int32)

decoded_preds  = [p.strip() for p in bart_tokenizer.batch_decode(predictions, skip_special_tokens=True)]
decoded_labels = [l.strip() for l in bart_tokenizer.batch_decode(labels, skip_special_tokens=True)]

rouge_result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
bleu_result  = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])

print("\nRESULTADOS FINALES EN TEST")
print(f"  ROUGE-1: {rouge_result['rouge1']:.4f}")
print(f"  ROUGE-2: {rouge_result['rouge2']:.4f}")
print(f"  ROUGE-L: {rouge_result['rougeL']:.4f}")
print(f"  BLEU:    {bleu_result['bleu']:.4f}")

# 10. GUARDAR

df_resultados = df_test.copy()
df_resultados["rag_input"]             = test_rag_inputs
df_resultados["generated_explanation"] = decoded_preds
df_resultados["reference_explanation"] = decoded_labels

results_file = os.path.join(RESULTS_PATH, "predictions_rag_bart.csv")
df_resultados.to_csv(results_file, index=False)
print(f"\nPredicciones guardadas en: {results_file}")

trainer.save_model(os.path.join(output_dir, "best_model"))
print(f"Modelo guardado en: {output_dir}/best_model")