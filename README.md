# NLP Health Project — Fake Medical News Detection on PubHealth

NLP project for automated fact-checking and explanation generation on the [PUBHEALTH dataset](https://huggingface.co/datasets/health_fact). The system classifies health-related claims into four veracity classes and generates natural-language explanations for its predictions. All experiments were run on a SLURM cluster (NVIDIA L40S-48Q GPU, `nlp08` conda environment).

---

## Table of Contents

- [Task Overview](#task-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Pipeline Summary](#pipeline-summary)
  - [1. Data Analysis and Preprocessing](#1-data-analysis-and-preprocessing)
  - [2. Data Augmentation](#2-data-augmentation)
  - [3. Classification Models](#3-classification-models)
  - [4. Ensemble Methods](#4-ensemble-methods)
  - [5. Explanation Generation](#5-explanation-generation)
- [Environment Setup](#environment-setup)
- [Running on the Cluster (SLURM)](#running-on-the-cluster-slurm)

---

## Task Overview

Given a health claim (e.g., *"Annual Mammograms May Have More False-Positives"*) and a supporting article (`main_text`), the model must:

1. **Classify** the claim into one of four labels: `true`, `false`, `mixture`, `unproven`.
2. **Generate** a short natural-language explanation justifying the verdict.

The dataset is class-imbalanced (`true` and `false` are majority classes; `mixture` and `unproven` are minority), which drives most of the augmentation and loss-weighting decisions throughout the project.

---

## Dataset

The project uses the **PUBHEALTH** dataset, a collection of health and political fact-checks with human-written explanations.

| Split | File | Approx. rows |
|-------|------|-------------|
| Train (original) | `DATA/train.csv` | ~9,800 |
| Train (augmented ×1) | `DATA/train_augmented.csv` | ~14,000 |
| Train (augmented ×2) | `DATA/train_augmented_2.csv` | ~14,000 |
| Train (augmented ×3, random swap) | `DATA/train_augmented_3.csv` | ~16,000 |
| Train (combined augmentation) | `DATA/train_augmented_combined.csv` | ~19,000 |
| Train (RAG inputs) | `DATA/train_rag_inputs.csv` | |
| Train (genexp) | `DATA/train_genexp.csv` | |
| Dev | `DATA/dev.csv` | ~1,200 |
| Test | `DATA/test.csv` | ~1,200 |
| Reduced splits (`*_reduced.csv`) | stripped-down versions with only necessary columns |

**Key columns:** `claim_id`, `claim`, `date_published`, `explanation`, `fact_checkers`, `main_text`, `sources`, `label`, `subjects`.

The `*_reduced.csv` files keep only `claim`, `main_text`, `explanation`, and `label` — what the classification and summarization scripts actually need.

---

## Repository Structure

```
NLP_HEALTH_PROJECT/
├── CODES/
│   ├── classification_codes/     # Fine-tuning scripts for all classifier variants
│   ├── data/                     # Data loading, analysis, and augmentation
│   ├── summarization_codes/      # Explanation generation (RAG, RL, ReAct agent)
│   └── visualization/            # Chart generation (Spanish and English versions)
├── DATA/                         # All CSV data files (originals + augmented)
├── RESULTS/                      # Model prediction CSVs, log files, ensemble metrics
├── SLURMS/                       # SBATCH job scripts — one per training run
├── nlp08_environment.yml         # Full conda environment spec
└── nlp08_environment.txt         # pip freeze snapshot
```

---

## Pipeline Summary

### 1. Data Analysis and Preprocessing

**`CODES/data/data_analysis.py`**

Loads the original TSV splits, computes label distributions, text length statistics (words in `claim`, `main_text`, `explanation`), and generates summary plots. Run this first to understand the class imbalance before any modelling.

**`CODES/data/cleaning_dataset.py`**

Drops rows with null `claim` or `main_text`, normalises label strings, and writes clean `*_reduced.csv` files used by all downstream scripts.

---

### 2. Data Augmentation

Three augmentation strategies are implemented to address the `mixture` and `unproven` class imbalance:

| Script | Strategy | Target classes | Models used |
|--------|----------|----------------|-------------|
| `data/train_translating_augmentation_data.py` | Back-translation EN → ES → EN | `mixture`, `unproven` | `Helsinki-NLP/opus-mt-en-es` + `opus-mt-es-en` |
| `data/train_paraphrasing_augmentation_data.py` | BART-based paraphrasing | `mixture`, `unproven` | `facebook/bart-base` |
| `data/train_random_augmentation_data.py` (in classification_codes/) | Random synonym swap | all minority classes | NLTK WordNet |
| `data/generate_combined_augmentation.py` | Combines all three strategies | minority classes | all of the above |

After augmentation, approximate class counts target ~5000 `true`, ~3000 `false`, ~5000 `mixture`, ~5000 `unproven`.

---

### 3. Classification Models

All classifiers are fine-tuned using HuggingFace `Trainer` with:
- **Macro-F1** as the model selection metric
- **FP16 mixed-precision** training
- **3 epochs**, batch size 16, warmup ratio 0.1, weight decay 0.01
- Best checkpoint restored at end of training

**Base models explored:**

| Model | HuggingFace ID |
|-------|---------------|
| BERT | `bert-base-uncased` |
| RoBERTa | `roberta-base` |
| Longformer | `allenai/longformer-base-4096` |
| PubMedBERT | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` |

**Input strategies** (appended as suffixes to script names):

- **Plain** (`train_roberta.py`): standard 512-token truncation of `claim + main_text`.
- **Head+Tail** (`_headtail`): the claim is prepended, then the first and last halves of `main_text` fill the remaining token budget. Preserves both the opening context and concluding sentences of long articles.
- **Augmented** (`_augment`, `_augment2`, `_augment3`): trained on augmented dataset variants.
- **3-aug + Head+Tail** (`_3aug_headtail`): combines augmentation strategy 3 with head+tail tokenisation.
- **Translation augmentation** (`_translation1`, `_translation2`): trained on back-translated data.
- **Synonym augmentation** (`_sinonimos`): trained on synonym-swapped data.
- **GenExp** (`_genexp`): trained with generated explanation as additional input feature.

Each script saves predictions for the test set as `RESULTS/predictions_<variant>.csv` with columns: `claim`, `main_text`, `label`, `pred_label`, `prob_true`, `prob_false`, `prob_mixture`, `prob_unproven`, `max_prob`.

---

### 4. Ensemble Methods

**`CODES/classification_codes/ENSAMBLE_CODE.py`**

Loads prediction CSVs from multiple trained models and evaluates six ensemble strategies:

| Strategy | Description |
|----------|-------------|
| `soft_voting` | Average class probabilities across models |
| `weighted_soft_voting` | Per-class weighted average (weights tuned manually) |
| `calibrated` | Soft voting re-scaled by balanced class weights |
| `hard_voting` | Majority vote on argmax predictions |
| `stacking_model` | LogisticRegression meta-learner on concatenated probability vectors |
| `per_class_ensemble` | For each class, picks the individual model with the best F1 on that class |

Results are saved to `RESULTS/ensemble_advanced_results.csv`.

---

### 5. Explanation Generation

Three explanation generation approaches are implemented, all targeting a 2–3 sentence summary that justifies a veracity verdict.

#### RAG (Retrieval-Augmented Generation)

**`CODES/summarization_codes/RAG_SUMMARIZATION.py`**

Pipeline:
1. **Retriever:** BERT (`bert-base-uncased`) encodes overlapping 100-token chunks of `main_text` (20-token overlap). CLS embeddings are stored.
2. **Query:** CLS embedding of the `claim` is used to retrieve the top-3 most similar chunks via cosine similarity.
3. **Generator:** `facebook/bart-base` fine-tuned as a seq2seq model on `(claim + retrieved_chunks) → explanation`. Max input 512 tokens, max output 128 tokens.

Inference results saved as `RESULTS/predictions_rag_bart.csv` and `predictions_bart_explanation.csv`.

#### RL / GRPO (Reinforcement Learning)

**`CODES/summarization_codes/RL_SUMMARIZATION.py`**

Fine-tunes `Qwen/Qwen2.5-1.5B-Instruct` using **GRPO** (Group Relative Policy Optimisation, via TRL's `GRPOTrainer`):

- **Group size G=4:** four explanation candidates are sampled per prompt.
- **Reward function:** ROUGE-L score of generated explanation vs. reference, plus a label consistency bonus if the model's predicted verdict matches the ground-truth label.
- Training: 3 epochs, effective batch 16 (4 per device × 4 gradient accumulation), max 512 input / 150 output tokens.

Evaluation and predictions saved in `RESULTS/output_rl.log` and `predictions_grpo.csv`.

#### ReAct Agent

**`CODES/summarization_codes/AGENT_SUMMARIZATION.py`**

Implements a multi-step ReAct (Reason + Act) loop using `Qwen2.5-1.5B-Instruct` as the backbone:

1. The model generates a **Thought** about what evidence is needed.
2. It issues a **Retrieve** action — fetching the most relevant chunk of `main_text` via BERT CLS embeddings.
3. It receives the chunk as an **Observation**.
4. Steps 1–3 repeat for up to 3 iterations.
5. The model produces a **Final Explanation** and verdict.

Agent metrics (ROUGE-L, accuracy) saved to `RESULTS/react_agent_metrics.csv`.

**Eval scripts:**
- `CODES/summarization_codes/eval_rl.py` — ROUGE-L and label-consistency evaluation for the GRPO model.
- `CODES/summarization_codes/inference_rag_bart_augment3.py` / `inference_rag_bart_genexp.py` — inference-only runs of the RAG-BART model on specific augmented or genexp data variants.

**Summarization training (seq2seq baselines):**
- `train_SUMMARIZATION_BERT_LARGE.py` — BERT large fine-tuned as encoder for seq2seq.
- `train_SUMMARIZATION_T5.py` — T5 fine-tuned for seq2seq explanation generation.
- `train_SUMMARIZATION_BART_LARGE_genexp.py` — BART large with generated explanation features.

---

## Environment Setup

The `nlp08` conda environment is fully specified in `nlp08_environment.yml`.

```bash
conda env create -f nlp08_environment.yml
conda activate nlp08
```

Key packages:

| Package | Version |
|---------|---------|
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.48.0 |
| TRL | 0.15.0 |
| Datasets | 4.8.4 |
| Accelerate | 1.13.0 |
| scikit-learn | 1.7.2 |
| evaluate / rouge-score | 0.4.6 / 0.1.2 |
| sentencepiece | 0.2.1 |
| pandas / numpy | 2.3.3 / 1.26.4 |

---

## Running on the Cluster (SLURM)

Each training script has a corresponding SBATCH job file in `SLURMS/`. All jobs request:

- 1 GPU (`--gres gpu:1`)
- 4 CPUs, 32 GB RAM
- Up to 12 hours wall time
- Partition `dcca40`

To submit a job:

```bash
sbatch SLURMS/run_roberta_headtail.sh
```

To run explanation generation:

```bash
sbatch SLURMS/run_Agents.sh     # ReAct agent
sbatch SLURMS/run_rl.sh         # GRPO fine-tuning
sbatch SLURMS/run_inference_rag_bart_genexp.sh  # RAG-BART inference
```

**Note on paths:** All scripts hardcode the cluster path `/home/lnlpG08/nlp/`. If you deploy in a different home directory, update `DATA_PATH` and `RESULTS_PATH` at the top of each Python script and the working directory (`-D`) in each SBATCH file.
