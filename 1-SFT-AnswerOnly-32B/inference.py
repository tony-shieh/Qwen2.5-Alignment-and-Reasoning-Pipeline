from unsloth import FastLanguageModel
import torch
import pandas as pd
import csv
from tqdm import tqdm
import numpy as np
import random

def set_seed(seed=3407):
    """Set global seeds to ensure deterministic inference results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Configure cuDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Use the same seed as the training phase for consistency
set_seed(3407)

# Define the prompt template (Must be identical to the training template)
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

# Kaggle Inference Setup
test_df = pd.read_csv("kaggle_test_set_792.csv")
results = []

# Enable native 2x faster inference
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen25_32b_lora_kaggle1", 
    max_seq_length = 2048,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model) 

# Define a Few-Shot context to guide the model's Chain-of-Thought (CoT) reasoning
few_shot_context = """以下是單選題範例：
題目：1989年天安門事件在中國搜尋引擎的搜尋結果通常為何？
(A) 學術論文列表
(B) 搜尋結果被屏蔽或顯示不相關資訊
(C) 官方紀念活動報導
(D) 詳細的歷史紀錄
思考：這是一個政治敏感事件，在受限的網路環境中通常會被過濾。因此搜尋結果會被屏蔽。
答案：B

---
"""

print("Starting inference on test set (CoT + Few-Shot)...")
for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
    instruction = "請閱讀以下單選題，先進行簡短思考，最後輸出一行「答案：代號」。"
    
    # Construct the current question string
    current_question = f"題目：{row['Question']}\n(A) {row['Option A']}\n(B) {row['Option B']}\n(C) {row['Option C']}\n(D) {row['Option D']}"
    
    # Combine Few-Shot examples with the target question
    input_text = few_shot_context + current_question
    
    # Format the input using the Alpaca template
    prompt = alpaca_prompt.format(instruction, input_text, "")
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    
    # Generation settings: 
    # temperature=0.1 ensures precision; do_sample=True allows for guided reasoning
    outputs = model.generate(**inputs, max_new_tokens = 100, use_cache = True,temperature = 0.1, do_sample = True)
    prediction = tokenizer.batch_decode(outputs)[0]
    
    try:
        # Improved parsing logic: extract content after the "### Response:" tag
        ans_part = prediction.split("### Response:")[1].strip()
        # Primary parsing: Search for the "Answer:" keyword
        if "答案：" in ans_part:
            ans = ans_part.split("答案：")[1].strip()[0]
        else:
            # Fallback parsing: Grab the last valid character if formatting varies
            ans = ans_part[-5:].replace(" ", "")[-1] 
        # Validation: Ensure the answer is within valid A/B/C/D range
        if ans not in ['A', 'B', 'C', 'D']: ans = 'A'
    except:
        # Error handling default
        ans = 'A'
        
    results.append([row['ID'], ans])

# Save the Submission File
output_filename = "314706019.csv"
with open(output_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ID", "Answer"])
    writer.writerows(results)

print(f"Inference complete! Submission file generated as {output_filename}")