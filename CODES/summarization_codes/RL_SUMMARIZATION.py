# GRPO-based explanation generation for PubHealth
# Policy: Qwen2.5-1.5B-Instruct fine-tuned with GRPO
# Reward: ROUGE-L vs reference explanation + label consistency bonus
# Based on: DeepSeekMath GRPO (Shao et al. 2024)

import os
import re
import numpy as np
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer
from trl import GRPOConfig
from datasets import Dataset

# 0. CONFIGURACIÓN

QWEN_PATH = "Qwen/Qwen2.5-1.5B-Instruct"  # Hugging Face Hub path

DATA_PATH    = "/home/lnlpG08/nlp/DATA/"
RESULTS_PATH = "/home/lnlpG08/nlp/RESULTS/grpo_explanation/"
os.makedirs(RESULTS_PATH, exist_ok=True)

EPOCHS         = 3
BATCH_SIZE     = 4    # per device
GRAD_ACCUM     = 4    # effective batch = 16
MAX_INPUT      = 512
MAX_NEW_TOKENS = 150
GROUP_SIZE     = 4    # G in GRPO: number of responses per prompt
LR             = 1e-5

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Group size G={GROUP_SIZE} | LR={LR}")

# 1. CARGAR DATOS

df_train = pd.read_csv(DATA_PATH + "train_reduced.csv")
df_dev   = pd.read_csv(DATA_PATH + "dev_reduced.csv")
df_test  = pd.read_csv(DATA_PATH + "test_reduced.csv")

for df, nombre in [(df_train, "train"), (df_dev, "dev"), (df_test, "test")]:
    df.dropna(subset=["claim", "main_text", "explanation", "label"], inplace=True)

print(f"Train: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")

# 2. BUILD PROMPTS
# Each training sample becomes a prompt that the policy must respond to

SYSTEM_PROMPT = """You are a medical fact-checking assistant.
Given a health claim and article evidence, write a concise explanation (2-3 sentences) of whether the claim is true, false, mixture (partially true), or unproven.
Base your explanation strictly on the provided evidence.
End your response with: Verdict: [true/false/mixture/unproven]"""


def build_prompt(claim, main_text):
    # Truncate main_text to avoid exceeding context
    words     = main_text.split()
    truncated = " ".join(words[:300])
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Claim: {claim}\n\n"
        f"Article evidence: {truncated}\n\n"
        f"Explanation:"
    )


def extract_verdict(text):
    """Extract predicted label from model output."""
    text_lower = text.lower()
    match = re.search(r'verdict:\s*(true|false|mixture|unproven)', text_lower)
    if match:
        return match.group(1)
    # Fallback: look for label anywhere in text
    for label in ["unproven", "mixture", "false", "true"]:
        if label in text_lower:
            return label
    return None


# 3. PREPARE DATASETS FOR GRPO
# GRPOTrainer expects dataset with "prompt" column
# and optionally metadata columns for reward computation

def make_grpo_dataset(df):
    records = []
    for _, row in df.iterrows():
        records.append({
            "prompt":      build_prompt(row["claim"], row["main_text"]),
            "reference":   row["explanation"],
            "label":       row["label"],
            "claim":       row["claim"],
        })
    return Dataset.from_list(records)


train_dataset = make_grpo_dataset(df_train)
dev_dataset   = make_grpo_dataset(df_dev)

print(f"\nGRPO train samples: {len(train_dataset)}")
print(f"Example prompt:\n{train_dataset[0]['prompt'][:300]}...")

# 4. REWARD FUNCTIONS
# Following DeepSeek R1 pattern: rule-based, verifiable rewards

from rouge_score import rouge_scorer as _rouge_scorer
_scorer = _rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def compute_rouge_l(prediction, reference):
    """Compute ROUGE-L between prediction and reference."""
    if not prediction.strip() or not reference.strip():
        return 0.0
    scores = _scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


def reward_function(completions, prompts=None, **kwargs):
    """
    GRPO reward function combining:
    1. ROUGE-L vs reference explanation (0 to 1)
    2. Label consistency bonus (+0.3 if verdict matches ground truth)
    3. Format bonus (+0.1 if response contains "Verdict:")
    4. Length penalty (penalize very short or very long responses)

    Returns list of scalar rewards, one per completion.
    """
    references = kwargs.get("reference", [""] * len(completions))
    labels     = kwargs.get("label",     [""] * len(completions))

    rewards = []
    for completion, reference, label in zip(completions, references, labels):

        # Extract text from completion
        if isinstance(completion, list):
            text = completion[0].get("content", "") if completion else ""
        else:
            text = str(completion)

        reward = 0.0

        # 1. ROUGE-L reward (main signal)
        rouge_score = compute_rouge_l(text, reference)
        reward     += rouge_score  # 0 to 1

        # 2. Label consistency bonus
        predicted_label = extract_verdict(text)
        if predicted_label is not None and predicted_label == str(label).lower():
            reward += 0.3
        elif predicted_label is None:
            reward -= 0.1  # penalize missing verdict

        # 3. Format bonus
        if "verdict:" in text.lower():
            reward += 0.1

        # 4. Length penalty
        words = text.split()
        if len(words) < 10:
            reward -= 0.3   # too short
        elif len(words) > 200:
            reward -= 0.1   # too long

        rewards.append(float(reward))

    return rewards


# 5. LOAD MODEL AND TOKENIZER

print("\nLoading Qwen2.5-1.5B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    QWEN_PATH,
    torch_dtype=torch.bfloat16,  # bfloat16 more stable than float16 for RL
    device_map="auto"
)

# 6. GRPO CONFIG

grpo_config = GRPOConfig(
    # Core GRPO
    num_generations=GROUP_SIZE,          # G: responses per prompt
    max_completion_length=MAX_NEW_TOKENS,
    temperature=0.7,                     # sampling temperature for group generation
    beta=0.01,                           # KL penalty coefficient

    # Training
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=1.0,
    bf16=True,

    # Eval and saving
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    save_total_limit=1,
    logging_steps=50,

    # Output
    output_dir=os.path.join(RESULTS_PATH, "grpo_model"),
    report_to="none",

    # Prompt truncation
)

# 7. GRPO TRAINER

print("\nInitializing GRPOTrainer...")
trainer = GRPOTrainer(
    model=model,
    args=grpo_config,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    reward_funcs=reward_function,
    processing_class=tokenizer,
)

# 8. TRAIN

print("\nEmpezando GRPO training...")
print(f"Each prompt generates {GROUP_SIZE} responses, rewards computed per group")
print(f"Advantage = (reward - mean(group_rewards)) / std(group_rewards)")
trainer.train(resume_from_checkpoint="/home/lnlpG08/nlp/RESULTS/grpo_explanation/grpo_model/checkpoint-6000")

# Save model
trainer.save_model(os.path.join(RESULTS_PATH, "grpo_model", "best_model"))
print("Model saved.")

# 9. INFERENCE ON TEST SET

print("\nRunning inference on test set...")
model.eval()


def generate_explanation(claim, main_text, max_new_tokens=MAX_NEW_TOKENS):
    prompt = build_prompt(claim, main_text)
    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=MAX_INPUT
    ).to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


generated_explanations = []
for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc="Test inference"):
    explanation = generate_explanation(row["claim"], row["main_text"])
    generated_explanations.append(explanation)

# 10. EVALUATE

print("\nComputing metrics...")
references = df_test["explanation"].tolist()

