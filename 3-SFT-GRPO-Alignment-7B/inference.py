import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch, pandas as pd, re, collections
from tqdm import tqdm
from unsloth import FastLanguageModel
import warnings
from transformers import logging as hf_logging
warnings.filterwarnings("ignore")
hf_logging.set_verbosity_error()

LORA_MODEL_PATH = "qwen-lora" 
SUBMISSION_FILE = "submission.csv"

print(f"Loading ultimate two-stage model ({LORA_MODEL_PATH})")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=LORA_MODEL_PATH, max_seq_length=2048, load_in_4bit=True,
)
FastLanguageModel.for_inference(model)
tokenizer.pad_token = tokenizer.eos_token

# Clean, single prompt
PROMPT = """你是一位具備頂尖邏輯的AI。請嚴密分析選項，使用逐步排除法。
【嚴格輸出格式】
你必須「精準」遵守以下XML格式，絕不能省略標籤：
<reasoning>
寫下你的逐步推論過程...
</reasoning>
<answer>單一正確字母(A,B,C或D)</answer>"""

df_test = pd.read_csv("dataset/kaggle_test_set_792.csv")
predictions = []
failed_count = 0

print(f"\nStarting test set inference")
for index, row in tqdm(df_test.iterrows(), total=len(df_test)):
    question = row['Question']
    opts = f"A. {row['Option A']}\nB. {row['Option B']}\nC. {row['Option C']}\nD. {row['Option D']}"
    prompt_text = f"題目：{question}\n{opts}\n請嚴密推理並選出正確答案。"

    extracted_answers = []
    
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": prompt_text}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")

    outputs = model.generate(
        input_ids=inputs, 
        max_new_tokens=384,  
        use_cache=True,
        temperature=0.2,     
        top_p=0.9, 
        do_sample=True,
        num_return_sequences=3, # 3-way majority vote
        pad_token_id=tokenizer.eos_token_id,
    )

    responses = tokenizer.batch_decode(outputs[:, inputs.shape[1]:], skip_special_tokens=True)
    for resp in responses:
        match = re.search(r'<answer>\s*([A-D])\s*</answer>', resp)
        if match: 
            extracted_answers.append(match.group(1))
    
    if not extracted_answers:
        final_answer = "B" 
        failed_count += 1
    else:
        counter = collections.Counter(extracted_answers)
        final_answer = counter.most_common(1)[0][0]
    predictions.append(final_answer)

print("\nChecking prediction distribution")
distribution = collections.Counter(predictions)
total_preds = len(predictions)
for ans, count in distribution.most_common():
    print(f"選項 {ans}: {count} 題 ({count/total_preds*100:.1f}%)")

print(f"Format parsing failure count: {failed_count} questions")

df_submission = pd.DataFrame({'ID': df_test['ID'], 'Answer': predictions})
df_submission.to_csv(SUBMISSION_FILE, index=False)

with open("distribution_9_final.txt", "w") as f:
    for k, v in distribution.items():
        f.write(f"Option {k}: {v} questions\n")
    f.write(f"Parsing failures: {failed_count} questions\n")

print(f"inference complete")