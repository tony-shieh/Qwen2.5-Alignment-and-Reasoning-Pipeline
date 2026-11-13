import os
# Prevent OOM: Enable PyTorch memory allocator configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import pandas as pd
from datasets import Dataset
import wandb
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments

# Initialize WandB
wandb.init(project="DeepLearing_HW2_Reasoning_LLM", name="Qwen-14B-SFT-V10-OOM_Fixed")

max_seq_length = 4096 
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-14B-Instruct-bnb-4bit", 
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

# Process Data
df = pd.read_csv("data/HW2_.csv")
SYSTEM_PROMPT = "你是一個專業且中立的選擇題解題專家。請針對題目進行嚴密的邏輯推理，客觀分析每個選項，最後明確給出你的答案。"

def format_chatml(row):
    instruction = f"題目：{row['題目']}\n選項A：{row['選項A']}\n選項B：{row['選項B']}\n選項C：{row['選項C']}\n選項D：{row['選項D']}\n請給出推理過程並選出正確答案。"
    response = f"{row['推理回答過程']}"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}

df_formatted = df.apply(format_chatml, axis=1, result_type="expand")
train_dataset = Dataset.from_pandas(df_formatted)

# LoRA Configuration
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,              
    lora_alpha = 64,     
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
)

# Training Configuration
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 4,
    args = TrainingArguments(
        per_device_train_batch_size = 1,  # Reduced from 4 to 1 to significantly lower VRAM consumption
        gradient_accumulation_steps = 16, # Increased to 16 to ensure the effective Batch Size remains 16 (1 * 16 = 16)
        warmup_ratio = 0.1,              
        num_train_epochs = 2,            
        learning_rate = 1e-4,            
        lr_scheduler_type = "cosine",    
        seed = 3407,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(), 
        logging_steps = 5,               
        optim = "adamw_8bit",
        weight_decay = 0.01,             
        output_dir = "outputs_v10_14B",  
        report_to = "wandb",
    ),
)

trainer.train()

# Save Model
model.save_pretrained("Qwen2.5-14B-LoRA-NYCU-DL-HW2")
tokenizer.save_pretrained("Qwen2.5-14B-LoRA-NYCU-DL-HW2")
print("model saved successfully")