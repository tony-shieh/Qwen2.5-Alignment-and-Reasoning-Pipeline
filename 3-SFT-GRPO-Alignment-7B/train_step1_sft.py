import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import pandas as pd
from datasets import Dataset
import wandb
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

wandb.login()
wandb.init(project="HW5-Kaggle3-Qwen", name="5090-Step1-SFT-Cleaneddata-v2")

SYSTEM_PROMPT = """你是一位具備頂尖邏輯的AI。請嚴密分析選項，使用逐步排除法。
【嚴格輸出格式】
你必須「精準」遵守以下XML格式，絕不能省略標籤：
<reasoning>
寫下你的逐步推論過程...
</reasoning>
<answer>單一正確字母(A,B,C或D)</answer>"""

print("Loading base model for SFT")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=64, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth", random_state=3407,
)

df_train = pd.read_csv("dataset/HW2_cleaned.csv")

def format_sft(row):
    question = row['題目']
    opts = f"A. {row['選項A']}\nB. {row['選項B']}\nC. {row['選項C']}\nD. {row['選項D']}"
    prompt = f"題目：{question}\n{opts}\n請嚴密推理並選出正確答案。"
    
    # Directly use the correct reasoning process from the dataset to teach the model
    reasoning_text = str(row['推理回答過程']) if '推理回答過程' in row and pd.notna(row['推理回答過程']) else "進行邏輯推導。"
    answer_text = str(row['答案']).strip()
    
    response = f"<reasoning>\n{reasoning_text}\n</reasoning>\n<answer>{answer_text}</answer>"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

# Added remove_columns=df_train.columns.tolist() to completely eliminate data type conflict landmines
train_dataset = Dataset.from_pandas(df_train).map(format_sft, remove_columns=df_train.columns.tolist())

trainer = SFTTrainer(
    model=model, tokenizer=tokenizer, train_dataset=train_dataset, dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        warmup_ratio=0.1, num_train_epochs=2, learning_rate=5e-5,
        lr_scheduler_type="cosine", logging_steps=1,
        disable_tqdm=False,
        fp16=not torch.cuda.is_bf16_supported(), bf16=torch.cuda.is_bf16_supported(),
        output_dir="outputs_sft", report_to="wandb",
    ),
)
print("Starting SFT training (Building baseline knowledge)")
trainer.train()
model.save_pretrained("qwen-step1-sft-lora-cleaneddata")
tokenizer.save_pretrained("qwen-step1-sft-lora-cleaneddata")
wandb.finish()
print("SFT complete! The model has learned the perfect reasoning format")