# ReAct Agent for explanation generation on PubHealth
# Pipeline: Thought → Action (retrieve chunk) → Observation → repeat → Final Explanation
# Backbone: Qwen2.5-1.5B-Instruct (instruction-tuned, cached)
# Retriever: BERT CLS embeddings (cached)

import os
import numpy as np
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from transformers import (
    AutoTokenizer, AutoModel,
    AutoModelForCausalLM,
)

# 0. CONFIGURACIÓN

QWEN_PATH = "Qwen/Qwen2.5-1.5B-Instruct"
BERT_PATH = "bert-base-uncased"  # Usamos un BERT preentrenado estándar para el retriever
DATA_PATH    = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/RESULTS/react_agent_explanation/"
os.makedirs(RESULTS_PATH, exist_ok=True)

CHUNK_SIZE    = 100
CHUNK_OVERLAP = 20
TOP_K         = 3
MAX_NEW_TOKENS = 200
MAX_STEPS     = 3   # max ReAct iterations per sample

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# 1. CARGAR DATOS

df_train = pd.read_csv(DATA_PATH + "train_reduced.csv")
df_dev   = pd.read_csv(DATA_PATH + "dev_reduced.csv")
df_test  = pd.read_csv(DATA_PATH + "test_reduced.csv")

for df, nombre in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
    df.dropna(subset=["claim", "main_text", "explanation"], inplace=True)

print(f"Train: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

# 2. BERT RETRIEVER

print("\nLoading BERT retriever...")
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
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_decoded = bert_tokenizer.decode(tokens[start:end], skip_special_tokens=True)
        chunks.append(chunk_decoded)
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def retrieve_chunks(claim, main_text, k=3, query_suffix=""):
    """Retrieve top-k chunks. query_suffix allows the agent to refine the query."""
    query  = claim + " " + query_suffix if query_suffix else claim
    chunks = chunk_text(main_text, CHUNK_SIZE, CHUNK_OVERLAP)

    if len(chunks) <= k:
        return chunks, list(range(len(chunks)))

    claim_emb   = get_cls_embedding(query)
    chunk_embs  = np.array([get_cls_embedding(c) for c in chunks])
    claim_norm  = claim_emb  / (np.linalg.norm(claim_emb)  + 1e-8)
    chunks_norm = chunk_embs / (np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-8)
    scores      = chunks_norm @ claim_norm
    top_idx     = sorted(np.argsort(scores)[-k:].tolist())
    return [chunks[i] for i in top_idx], top_idx


# 3. QWEN AGENT

print("\nLoading Qwen2.5-1.5B-Instruct agent...")
qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
qwen_model     = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)
qwen_model.eval()


def qwen_generate(prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate text with Qwen given a prompt string."""
    inputs = qwen_tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=2048
    ).to(device)
    with torch.no_grad():
        output = qwen_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=qwen_tokenizer.eos_token_id
        )
    # Decode only the newly generated tokens
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return qwen_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# 4. REACT AGENT PROMPT TEMPLATES

SYSTEM_PROMPT = """You are a medical fact-checking assistant. Given a health claim and evidence from an article, your task is to write a brief explanation of whether the claim is true, false, mixture (partially true), or unproven.

You must follow this reasoning format:
Thought: reason about what you know and what evidence you need
Action: retrieve[query] — retrieve more evidence using a specific query
Observation: the retrieved evidence will be provided
... (repeat Thought/Action/Observation as needed, maximum {max_steps} times)
Thought: I now have enough evidence to write the explanation
Final Answer: write a concise explanation (2-3 sentences) of the claim verdict

Available actions:
- retrieve[query]: search the article for evidence related to the query
- finish[explanation]: provide the final explanation

Important: always end with "Final Answer:" followed by your explanation.
""".format(max_steps=MAX_STEPS)


def build_initial_prompt(claim, first_observation):
    return f"""{SYSTEM_PROMPT}

Claim: {claim}

Thought: I need to find evidence about this claim in the article.
Action: retrieve[{claim}]
Observation: {first_observation}
"""


def parse_action(text):
    """Extract action type and argument from generated text."""
    text = text.strip()
    if "retrieve[" in text:
        start = text.index("retrieve[") + len("retrieve[")
        end   = text.index("]", start) if "]" in text[start:] else len(text)
        query = text[start:end].strip()
        return "retrieve", query
    if "Final Answer:" in text:
        answer = text.split("Final Answer:")[-1].strip()
        return "finish", answer
    if "finish[" in text:
        start  = text.index("finish[") + len("finish[")
        end    = text.index("]", start) if "]" in text[start:] else len(text)
        answer = text[start:end].strip()
        return "finish", answer
    # If no valid action found, treat remaining text as final answer
    return "finish", text


# 5. REACT LOOP

# 5. REACT LOOP

def run_react_agent(claim, main_text, max_steps=MAX_STEPS):
    """
    Run a ReAct-style retrieval + reasoning loop.

    Returns:
        explanation (str)
        trajectory (str)  # full Thought/Action/Observation trace
    """

    # Initial retrieval
    retrieved_chunks, _ = retrieve_chunks(
        claim,
        main_text,
        k=TOP_K
    )

    observation = "\n".join(retrieved_chunks)

    prompt = build_initial_prompt(claim, observation)

    trajectory = prompt

    for step in range(max_steps):

        generation = qwen_generate(prompt)

        trajectory += generation + "\n"

        action_type, action_arg = parse_action(generation)

        # Agent finished
        if action_type == "finish":
            return action_arg.strip(), trajectory

        # Agent requests retrieval
        if action_type == "retrieve":

            retrieved_chunks, idxs = retrieve_chunks(
                claim,
                main_text,
                k=TOP_K,
                query_suffix=action_arg
            )

            observation = "\n".join(retrieved_chunks)

            prompt = (
                trajectory
                + f"\nObservation: {observation}\n"
            )

            trajectory = prompt

    # Safety fallback if max steps reached
    fallback_prompt = (
        trajectory
        + "\nThought: I now have enough evidence to write the explanation.\n"
        + "Final Answer:"
    )

    final_answer = qwen_generate(
        fallback_prompt,
        max_new_tokens=100
    )

    return final_answer.strip(), trajectory


# 6. GENERATE EXPLANATIONS

print("\nGenerating explanations...")

predictions = []
references = []
trajectories = []

for _, row in tqdm(df_test.iterrows(), total=len(df_test)):

    claim = row["claim"]
    main_text = row["main_text"]
    gold_explanation = row["explanation"]

    try:
        pred_explanation, trace = run_react_agent(
            claim,
            main_text
        )

    except Exception as e:
        print(f"Error: {e}")
        pred_explanation = ""
        trace = ""

    predictions.append(pred_explanation)
    references.append(gold_explanation)
    trajectories.append(trace)


# Save generations

results_df = pd.DataFrame({
    "claim": df_test["claim"].tolist(),
    "gold_explanation": references,
    "pred_explanation": predictions,
    "trajectory": trajectories
})

results_file = os.path.join(
    RESULTS_PATH,
    "react_agent_predictions.csv"
)

results_df.to_csv(results_file, index=False)

print(f"\nSaved predictions to: {results_file}")


# 7. EVALUATION

print("\nEvaluating...")

# Load saved predictions
results_df = pd.read_csv(os.path.join(RESULTS_PATH, "react_agent_predictions.csv"))
results_df.dropna(subset=["pred_explanation", "gold_explanation"], inplace=True)

predictions = results_df["pred_explanation"].tolist()
references  = results_df["gold_explanation"].tolist()

print(f"Evaluating {len(predictions)} samples...")

# ROUGE
rouge = evaluate.load("rouge")
rouge_scores = rouge.compute(predictions=predictions, references=references)

# BERTScore
bertscore = evaluate.load("bertscore")
bert_scores = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en"
)
# BLEU
bleu = evaluate.load("bleu")
bleu_scores = bleu.compute(
    predictions=predictions,
    references=[[r] for r in references]  # bleu expects list of references per sample
)

metrics = {
    "ROUGE-1": rouge_scores["rouge1"],
    "ROUGE-2": rouge_scores["rouge2"],
    "ROUGE-L": rouge_scores["rougeL"],
    "BLEU": bleu_scores["bleu"],
    "BERTScore-Precision": float(np.mean(bert_scores["precision"])),
    "BERTScore-Recall": float(np.mean(bert_scores["recall"])),
    "BERTScore-F1": float(np.mean(bert_scores["f1"]))
}

for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

metrics_df = pd.DataFrame([metrics])

metrics_file = os.path.join(
    RESULTS_PATH,
    "react_agent_metrics.csv"
)

metrics_df.to_csv(metrics_file, index=False)

print(f"\nSaved metrics to: {metrics_file}")


# 8. EXAMPLE OUTPUTS

print("\n===== SAMPLE PREDICTIONS =====")

for i in range(min(5, len(results_df))):

    print("\n----------------------------------")
    print("CLAIM:")
    print(results_df.iloc[i]["claim"])

    print("\nGOLD:")
    print(results_df.iloc[i]["gold_explanation"])

    print("\nPRED:")
    print(results_df.iloc[i]["pred_explanation"])