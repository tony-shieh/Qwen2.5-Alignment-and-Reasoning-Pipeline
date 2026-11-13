from unsloth import FastLanguageModel
import torch
import pandas as pd
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
import gc
import numpy as np
import random

def set_seed(seed=3407):
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in Cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use 3407, a common seed for Unsloth
set_seed(3407) 

# Model Configuration
max_seq_length = 2048
dtype = None 
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

#Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  
    loftq_config = None, 
)

# Standard Alpaca-style prompt template
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Essential to prevent infinite generation
EOS_TOKEN = tokenizer.eos_token 

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Format strings and append EOS_TOKEN
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

# Load the integrated training dataset
df = pd.read_csv("HW1_.csv")

# Data conversion function for Alpaca format
def convert_to_alpaca(row):
    # Instruction: Clearly define the multiple-choice task
    instruction = "請閱讀以下單選題，並輸出正確答案的選項代號（A、B、C 或 D）。"
    
    # Input: Combine the question text with its four options
    input_text = f"題目：{row['題目']}\n(A) {row['選項A']}\n(B) {row['選項B']}\n(C) {row['選項C']}\n(D) {row['選項D']}"
    
    # Output: The correct answer label as a string
    return {
        "instruction": instruction,
        "input": input_text,
        "output": str(row['答案']) # ensure the answer is a string for consistent formatting
    }

# Transform data and create Hugging Face Dataset
formatted_data = [convert_to_alpaca(row) for _, row in df.iterrows()]
dataset = Dataset.from_list(formatted_data)

# Map the formatting function with batching enabled
dataset = dataset.map(formatting_prompts_func, batched = True)


from trl import SFTConfig, SFTTrainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    args = SFTConfig(
        per_device_train_batch_size = 2,      
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        num_train_epochs = 3,                
        max_steps = -1, # Set to -1 to run through all epochs                     
        learning_rate = 1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        output_dir = "outputs",
        report_to = "wandb", # Connect to your Weights & Biases account
    ),
)

# Show current memory statistics before training
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
# Execute the training process
trainer_stats = trainer.train()

# Post-training statistics and memory cleanup
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"訓練完成！耗時: {trainer_stats.metrics['train_runtime']} 秒")
print(f"最高顯存佔用: {used_memory} GB")

# Clean up memory for subsequent inference
del trainer
gc.collect()
torch.cuda.empty_cache()

# Save LoRA model weights and tokenizer for report submission
model.save_pretrained("qwen25_32b_lora_kaggle1")
tokenizer.save_pretrained("qwen25_32b_lora_kaggle1")

