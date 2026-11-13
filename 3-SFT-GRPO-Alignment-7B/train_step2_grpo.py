import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)
import torch, pandas as pd, re, wandb
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

wandb.login()
wandb.init(project="HW5-Kaggle3-Qwen", name="5090-Step2-GRPO-v9")

SYSTEM_PROMPT = """你是一位具備頂尖邏輯的AI。請嚴密分析選項，使用逐步排除法。
【嚴格輸出格式】
你必須「精準」遵守以下XML格式，絕不能省略標籤：
<reasoning>
寫下你的逐步推論過程...
</reasoning>
<answer>單一正確字母(A,B,C或D)</answer>"""

print("Loading the SFT trained model")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="qwen-step1-sft-lora-cleaneddata",
    max_seq_length=2048, load_in_4bit=True,
)
model.warnings_issued = {} 

def format_data(example):
    question = example['題目']
    opts = f"A. {example['選項A']}\nB. {example['選項B']}\nC. {example['選項C']}\nD. {example['選項D']}"
    prompt = f"題目：{question}\n{opts}\n請嚴密推理並選出正確答案。"
    return {
        "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "answer": str(example['答案']).strip()
    }

df_train = pd.read_csv("dataset/HW2_cleaned.csv").dropna(subset=['答案'])
train_dataset = Dataset.from_pandas(df_train).map(format_data, remove_columns=df_train.columns.tolist())

# Purest correctness reward (No class bias)
def correctness_reward(prompts, completions, answer, **kwargs) -> list[float]:
    rewards = []
    for comp, ans in zip(completions, answer):
        content = comp[0]["content"] if isinstance(comp, list) else comp
        ans_match = re.search(r'<answer>\s*([A-D])\s*</answer>', content)
        if ans_match and ans_match.group(1) == ans:
            rewards.append(4.0)
        elif ans_match and ans_match.group(1) != ans:
            rewards.append(-4.0)
        else:
            rewards.append(-3.0)
    return rewards

# Purest format reward
def format_reward(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for comp in completions:
        content = comp[0]["content"] if isinstance(comp, list) else comp
        good_format = re.search(r"<reasoning>.*?</reasoning>\s*<answer>[A-D]</answer>", content, re.DOTALL)
        rewards.append(1.0 if good_format else -1.0)
    return rewards

training_args = GRPOConfig(
    use_vllm=False, learning_rate=2e-6, 
    adam_beta1=0.9, adam_beta2=0.99, weight_decay=0.1,
    lr_scheduler_type="cosine", logging_steps=1, disable_tqdm=False,
    per_device_train_batch_size=1, gradient_accumulation_steps=4,
    num_generations=4,          # Increased to 4-way comparison for more precise gradients
    max_completion_length=384,  # Held at 384, avoiding cuts to 256 which would degrade logic generation
    max_steps=500,              
    num_train_epochs=1, save_steps=100, max_grad_norm=0.1,
    report_to="wandb", output_dir="outputs_grpo_v8",
)

trainer = GRPOTrainer(
    model=model, processing_class=tokenizer,
    reward_funcs=[correctness_reward, format_reward],
    args=training_args, train_dataset=train_dataset,
)

print("Starting GRPO training")
trainer.train()
model.save_pretrained("qwen-lora")
tokenizer.save_pretrained("qwen-lora")
wandb.finish()
print("model training complete")