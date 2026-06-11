# Inferencia con el modelo RAG-BART ya entrenado
# Genera las explicaciones para train_augmented_3.csv (random swap/deletion)
# Necesario para entrenar roberta_headtail_augment3_genexp (mejor modelo para UNPROVEN)
# Output: train_augmented_3_genexp.csv en DATA/
 
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
 
CHUNK_SIZE     = 100
CHUNK_OVERLAP  = 20
TOP_K          = 3
MAX_INPUT      = 512
MAX_NEW_TOKENS = 128
BATCH_SIZE     = 16
 
DATA_PATH = "/home/lnlpG08/nlp/DATA/"
 
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("\nCargando train_augmented_3.csv...")
df = pd.read_csv(DATA_PATH + "train_augmented_3.csv")
 
antes = len(df)
df.dropna(subset=["claim", "main_text"], inplace=True)
if len(df) < antes:
    print(f"  Eliminadas {antes - len(df)} filas con nulos")
 
# Solo nos interesan las clases del proyecto
df = df[df["label"].isin(["true", "false", "mixture", "unproven"])].copy()
df = df.reset_index(drop=True)
 
print(f"Total filas: {len(df)}")
print("Distribución:", df["label"].value_counts().to_dict())
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. BERT RETRIEVER
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
        end           = min(start + chunk_size, len(tokens))
        chunk_tokens  = tokens[start:end]
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
print("\nConstruyendo RAG inputs para train_augmented_3...")
rag_inputs = []
for _, row in tqdm(df.iterrows(), total=len(df), desc="RAG inputs"):
    retrieved  = retrieve_top_chunks(row["claim"], row["main_text"], k=TOP_K)
    rag_input  = f"claim: {row['claim']} evidence: {retrieved}"
    rag_inputs.append(rag_input)
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. CARGAR MODELO RAG-BART ENTRENADO
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nCargando modelo RAG-BART desde: {BART_MODEL_DIR}")
bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_DIR)
bart_model     = BartForConditionalGeneration.from_pretrained(BART_MODEL_DIR).to(device)
bart_model.eval()
print("Modelo cargado correctamente.")
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. GENERAR EXPLICACIONES
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerando explicaciones...")
all_explanations = []
for i in tqdm(range(0, len(rag_inputs), BATCH_SIZE), desc="Generando"):
    batch  = rag_inputs[i:i+BATCH_SIZE]
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
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. GUARDAR CSV
# ─────────────────────────────────────────────────────────────────────────────
df["generated_explanation"] = all_explanations
df["rag_input"]             = rag_inputs
 
out_path = DATA_PATH + "train_augmented_3_genexp.csv"
df.to_csv(out_path, index=False)
print(f"\n¡Listo! Guardado en: {out_path}")
print(f"Total filas: {len(df)}")