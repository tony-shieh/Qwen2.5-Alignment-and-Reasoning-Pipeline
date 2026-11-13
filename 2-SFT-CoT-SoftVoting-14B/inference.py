import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from unsloth import FastLanguageModel
import warnings
from transformers import logging as hf_logging

# Suppress annoying warning messages to keep the progress bar clean
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

print("Loading the 14B model")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen2.5-14B-LoRA-NYCU-DL-HW2", # Point to your trained 14B
    max_seq_length = 4096,             
    dtype = None,
    load_in_4bit = True 
)
FastLanguageModel.for_inference(model)

SYSTEM_PROMPT = "你是一個專業且中立的選擇題解題專家。請針對題目進行嚴密的邏輯推理，客觀分析每個選項，最後明確給出你的答案。"
output_filename = "submission_314706019.csv"

# Extract Token IDs for options A, B, C, D from the Qwen dictionary
token_ids = {
    "A": tokenizer.encode("A", add_special_tokens=False)[0],
    "B": tokenizer.encode("B", add_special_tokens=False)[0],
    "C": tokenizer.encode("C", add_special_tokens=False)[0],
    "D": tokenizer.encode("D", add_special_tokens=False)[0]
}

test_df = pd.read_csv("data/kaggle_test_set_792.csv")
results = []
processed_ids = set()

# Mechanism to resume from a checkpoint
if os.path.exists(output_filename):
    old_df = pd.read_csv(output_filename)
    results = old_df.to_dict('records')
    processed_ids = set(old_df["ID"].tolist())
    print(f"Found backup. Resuming from where it left off")

test_df_remaining = test_df[~test_df["ID"].isin(processed_ids)]

for index, row in tqdm(test_df_remaining.iterrows(), total=len(test_df_remaining), desc="14B 自洽性推論中"):
    instruction = f"題目：{row['Question']}\n選項A：{row['Option A']}\n選項B：{row['Option B']}\n選項C：{row['Option C']}\n選項D：{row['Option D']}\n請給出推理過程並選出正確答案。"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    input_length = inputs["input_ids"].shape[1] 
    
    # Dictionary to accumulate the 'confidence probability' across 3 inference iterations
    probs_sum = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
    
    # Core logic: Force 14B to evaluate the same question using 3 different reasoning paths
    for _ in range(3):
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1500, 
            use_cache=True, 
            do_sample=True,      # Enable sampling to explore different solution paths
            temperature=0.6,     # Slightly increase temperature for greater reasoning diversity
            top_p=0.9
        )
        
        reasoning_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        
        # Append the generated reasoning text and force the model to output the final choice
        final_prompt = prompt + reasoning_text + "\n綜合以上嚴密推理，本題的最終正確答案是選項："
        final_inputs = tokenizer([final_prompt], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            logits = model(**final_inputs).logits[:, -1, :] 
        
        # Extract the raw logits for the four options
        log_a = logits[0, token_ids["A"]]
        log_b = logits[0, token_ids["B"]]
        log_c = logits[0, token_ids["C"]]
        log_d = logits[0, token_ids["D"]]
        
        # Convert the logits of the four options into relative probabilities (0~1) using Softmax
        stacked_logits = torch.stack([log_a, log_b, log_c, log_d])
        probs = F.softmax(stacked_logits, dim=0)
        
        # Accumulate the probabilities for this iteration
        probs_sum["A"] += probs[0].item()
        probs_sum["B"] += probs[1].item()
        probs_sum["C"] += probs[2].item()
        probs_sum["D"] += probs[3].item()
    
    # # Select the answer with the highest accumulated probability after 3 iterations
    best_ans = max(probs_sum, key=probs_sum.get)
    results.append({"ID": row["ID"], "Answer": best_ans})

    # Save to file every 5 questions to prevent data loss
    if len(results) % 5 == 0 or len(results) == len(test_df):
        pd.DataFrame(results).to_csv(output_filename, index=False)

print(f"\n Inference completed! Results saved to {output_filename}")