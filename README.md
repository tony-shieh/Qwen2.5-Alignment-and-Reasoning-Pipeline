# Qwen2.5 Alignment & Reasoning Pipeline: From SFT to GRPO

**Author:** 謝宗穎 (Zong-Ying Shieh)  
**Affiliation:** Institute of Information Management, National Yang Ming Chiao Tung University (NYCU)  

This repository documents an end-to-end Large Language Model (LLM) alignment and reasoning optimization journey. Working under strict local-execution and open-source model constraints, this project explores how progressive fine-tuning and inference-time compute strategies enable smaller, parameter-efficient models (`32B` $\rightarrow$ `14B` $\rightarrow$ `7B`) to overcome refusal behaviors and solve complex Traditional Chinese multiple-choice reasoning tasks.

---

## 📊 Methodology & Performance Progression

| Stage / Folder | Base Model | Core Methodology | Inference Strategy | Hardware | Kaggle Score | Model Weights |
| :--- | :--- | :--- | :--- | :--- | :---: | :--- |
| **[01-SFT-AnswerOnly-32B](./01-SFT-AnswerOnly-32B)** | `Qwen2.5-32B-Instruct` (4-bit)[cite: 1] | Supervised Fine-Tuning (LoRA, $r=16$) on Answer-Only data[cite: 1] | 1-Shot CoT Prompting (`temp=0.1`)[cite: 1] | RTX 4090 (24GB)[cite: 1] | **0.73617**[cite: 1] | [Hugging Face](https://huggingface.co/tony-shieh/Qwen2.5-32B-LoRA-NYCU-DeepLearning-SFT-on-Answer-314706019)[cite: 1] |
| **[02-SFT-CoT-SoftVoting-14B](./02-SFT-CoT-SoftVoting-14B)** | `Qwen2.5-14B-Instruct` (4-bit)[cite: 2] | CoT SFT with OOM optimization (`grad_accum=16`, expandable segments)[cite: 2] | Logits Interception + 3-way Self-Consistency Softmax Voting[cite: 2] | RTX 5090 (32GB)[cite: 2] | **0.72340**[cite: 2] | [Hugging Face](https://huggingface.co/tony-shieh/Qwen2.5-14B-LoRA-NYCU-DL-HW2)[cite: 2] |
| **[03-SFT-GRPO-Alignment-7B](./03-SFT-GRPO-Alignment-7B)** | `Qwen2.5-7B-Instruct` (4-bit)[cite: 3] | **Two-Stage Pipeline**: XML-Schema SFT + **GRPO** Reinforcement Learning[cite: 3] | Low-temp (`0.2`) 3-way Majority Vote Parsing[cite: 3] | Local GPU[cite: 3] | **0.69787** | [SFT](https://huggingface.co/tony-shieh/Qwen2.5-7B-SFT-TW-QA) / [GRPO](https://huggingface.co/tony-shieh/Qwen2.5-7B-GRPO-TW-QA)[cite: 3] |

---

## 🛠️ Tech Stack & Engineering Highlights

* **Frameworks & Training:** `PyTorch`, `Transformers`, `Unsloth` (4-bit QLoRA acceleration), `TRL` (`SFTTrainer` & `GRPOTrainer`)[cite: 1, 3].
* **Inference-Time Compute:** Implemented direct neural network logits extraction to guarantee 100% format compliance, combined with Self-Consistency (SC) probability soft-voting[cite: 2].
* **Reinforcement Learning Alignment:** Utilized Group Relative Policy Optimization (GRPO) with custom reward functions to enforce strict `<reasoning>...</reasoning><answer>X</answer>` XML boundaries and logical accuracy without a separate critic model[cite: 3].
* **Reproducible Environment:** Managed across all stages using **`uv`** (`pyproject.toml`) for deterministic dependency resolution[cite: 1, 2, 3].