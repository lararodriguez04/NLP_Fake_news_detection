import os, torch, pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from rouge_score import rouge_scorer as _rouge_scorer
import evaluate

CHECKPOINT = "/home/lnlpG08/nlp/RESULTS/grpo_explanation/grpo_model/checkpoint-7353"
DATA_PATH  = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/RESULTS/grpo_explanation/"
MAX_NEW_TOKENS = 200
MAX_INPUT = 512
device = "cuda" if torch.cuda.is_available() else "cpu"

df_test = pd.read_csv(DATA_PATH + "test_reduced.csv")
df_test.dropna(subset=["claim", "main_text", "explanation"], inplace=True)
print(f"Test: {len(df_test)}")

print("Loading model from checkpoint...")
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, torch_dtype=torch.float16, device_map="auto")
model.eval()

SYSTEM = """You are a medical fact-checking assistant.
Given a health claim and article evidence, write a concise explanation (2-3 sentences) of whether the claim is true, false, mixture (partially true), or unproven.
Base your explanation strictly on the provided evidence.
End your response with: Verdict: [true/false/mixture/unproven]"""

def build_prompt(claim, main_text):
    return f"{SYSTEM}\n\nClaim: {claim}\n\nEvidence: {main_text[:1000]}\n\nExplanation:"

def generate_explanation(claim, main_text):
    prompt = build_prompt(claim, main_text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT).to(device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                temperature=1.0, pad_token_id=tokenizer.eos_token_id)
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

generated = []
for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Test inference"):
    generated.append(generate_explanation(row["claim"], row["main_text"]))

references = df_test["explanation"].tolist()

_scorer = _rouge_scorer.RougeScorer(["rouge1","rouge2","rougeL"], use_stemmer=True)
r1, r2, rl = [], [], []
for p, r in zip(generated, references):
    s = _scorer.score(r, p)
    r1.append(s["rouge1"].fmeasure)
    r2.append(s["rouge2"].fmeasure)
    rl.append(s["rougeL"].fmeasure)

bleu = evaluate.load("bleu")
bleu_score = bleu.compute(predictions=generated, references=[[r] for r in references])["bleu"]

print(f"\nRESULTADOS FINALES EN TEST")
print(f"ROUGE-1: {np.mean(r1):.4f}")
print(f"ROUGE-2: {np.mean(r2):.4f}")
print(f"ROUGE-L: {np.mean(rl):.4f}")
print(f"BLEU:    {bleu_score:.4f}")

df_test["generated_explanation"] = generated
df_test.to_csv(RESULTS_PATH + "predictions_grpo.csv", index=False)
print(f"\nGuardado en: {RESULTS_PATH}predictions_grpo.csv")
