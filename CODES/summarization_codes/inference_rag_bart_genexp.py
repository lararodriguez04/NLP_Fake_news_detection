
# Inferencia con el modelo RAG-BART ya entrenado
# Genera las explicaciones para train y dev (sin reentrenar nada)
# Output: train_full_genexp.csv y dev_genexp.csv en DATA/
 
import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, BartForConditionalGeneration, BartTokenizer
from tqdm import tqdm
 
# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
BERT_PATH      = "bert-base-uncased"
BART_MODEL_DIR = "/home/lnlpG08/nlp/RESULTS/rag_bart_explanation/rag-bart-explanation/best_model"
 
CHUNK_SIZE    = 100
CHUNK_OVERLAP = 20
TOP_K         = 3
MAX_INPUT     = 512
MAX_NEW_TOKENS = 128
BATCH_SIZE    = 16   # para la generación
 
DATA_PATH    = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/RESULTS/"
 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("\nCargando datos...")
df_train = pd.read_csv(DATA_PATH + "train.csv")
df_dev   = pd.read_csv(DATA_PATH + "dev_reduced.csv")
 
for df, nombre in [(df_train, "train"), (df_dev, "dev")]:
    antes = len(df)
    df.dropna(subset=["claim", "main_text"], inplace=True)
    if len(df) < antes:
        print(f"  {nombre}: eliminadas {antes - len(df)} filas con nulos")
 
print(f"Train: {len(df_train)} | Dev: {len(df_dev)}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. BERT RETRIEVER (igual que en el script original)
# ─────────────────────────────────────────────────────────────────────────────
print("\nCargando BERT retriever...")
bert_tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert_model     = AutoModel.from_pretrained(BERT_PATH).to(device)
bert_model.eval()
 
def get_cls_embedding(text, max_length=128):
    inputs = bert_tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=max_length,
        padding=True
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
 
def chunk_text(text, chunk_size=100, overlap=20):
    tokens = bert_tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start  = 0
    while start < len(tokens):
        end          = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_decoded = bert_tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_decoded)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks
 
def retrieve_top_chunks(claim, main_text, k=3):
    chunks = chunk_text(main_text, CHUNK_SIZE, CHUNK_OVERLAP)
    if len(chunks) == 0:
        return main_text[:500]
    if len(chunks) <= k:
        return " ".join(chunks)
    claim_emb   = get_cls_embedding(claim)
    chunk_embs  = np.array([get_cls_embedding(c) for c in chunks])
    claim_norm  = claim_emb  / (np.linalg.norm(claim_emb)  + 1e-8)
    chunks_norm = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    scores      = chunks_norm @ claim_norm
    top_indices = sorted(np.argsort(scores)[-k:].tolist())
    return " ".join([chunks[i] for i in top_indices])
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. CONSTRUIR RAG INPUTS
# ─────────────────────────────────────────────────────────────────────────────
def build_rag_inputs(df, split_name):
    rag_inputs = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"RAG inputs {split_name}"):
        retrieved  = retrieve_top_chunks(row["claim"], row["main_text"], k=TOP_K)
        rag_input  = f"claim: {row['claim']} evidence: {retrieved}"
        rag_inputs.append(rag_input)
    return rag_inputs
 
print("\nConstruyendo RAG inputs para train...")
train_rag_inputs = build_rag_inputs(df_train, "train")
 
print("\nConstruyendo RAG inputs para dev...")
dev_rag_inputs = build_rag_inputs(df_dev, "dev")
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. CARGAR MODELO RAG-BART ENTRENADO
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nCargando modelo RAG-BART desde: {BART_MODEL_DIR}")
bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_DIR)
bart_model     = BartForConditionalGeneration.from_pretrained(BART_MODEL_DIR).to(device)
bart_model.eval()
print("Modelo cargado correctamente.")
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERAR EXPLICACIONES (inferencia en batches)
# ─────────────────────────────────────────────────────────────────────────────
def generate_explanations(rag_inputs, split_name):
    all_explanations = []
    for i in tqdm(range(0, len(rag_inputs), BATCH_SIZE), desc=f"Generando {split_name}"):
        batch = rag_inputs[i:i+BATCH_SIZE]
        inputs = bart_tokenizer(
            batch,
            max_length=MAX_INPUT,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = bart_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=4,
                early_stopping=True
            )
        decoded = bart_tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_explanations.extend([d.strip() for d in decoded])
    return all_explanations
 
print("\nGenerando explicaciones para train...")
train_explanations = generate_explanations(train_rag_inputs, "train")
 
print("\nGenerando explicaciones para dev...")
dev_explanations = generate_explanations(dev_rag_inputs, "dev")
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. GUARDAR CSVs
# ─────────────────────────────────────────────────────────────────────────────
df_train_out = df_train.copy()
df_train_out["generated_explanation"] = train_explanations
df_train_out["rag_input"]             = train_rag_inputs
train_out_path = DATA_PATH + "train_full_genexp.csv"
df_train_out.to_csv(train_out_path, index=False)
print(f"\nTrain guardado en: {train_out_path}")
 
df_dev_out = df_dev.copy()
df_dev_out["generated_explanation"] = dev_explanations
df_dev_out["rag_input"]             = dev_rag_inputs
dev_out_path = DATA_PATH + "dev_genexp.csv"
df_dev_out.to_csv(dev_out_path, index=False)
print(f"Dev guardado en: {dev_out_path}")
 
print("\n¡Listo! Ya tienes train_full_genexp.csv y dev_genexp.csv en DATA/")
print(f"  train: {len(df_train_out)} filas")
print(f"  dev:   {len(df_dev_out)} filas")
 