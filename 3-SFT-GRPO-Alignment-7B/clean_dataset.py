import pandas as pd
import re

# Read the original dataset
df = pd.read_csv("dataset/HW2_.csv")

def get_answer_from_reasoning(text):
    text = str(text)
    # Search for all possible answer formats (e.g., "Answer: A", "Option B")
    matches = re.findall(r'(?:答案|最終答案|選項|選)\s*[：:]?\s*([ABCD])', text)
    if matches:
        return matches[-1] # Take the last occurrence found at the end of the reasoning
    return None

# Extract the answer from the reasoning process
df['推理得到的答案'] = df['推理回答過程'].apply(get_answer_from_reasoning)

# Filter: Keep only the rows where the "Answer" column matches the "Reasoning conclusion"
# Ensure the answer column has no extra spaces for accurate comparison
df['答案'] = df['答案'].astype(str).str.strip()
clean_df = df[df['答案'] == df['推理得到的答案'].astype(str).str.strip()].copy()

# Clean up redundant columns
clean_df = clean_df.drop(columns=['推理得到的答案'])

# Ensure the answer column in the output file also has no spaces
clean_df['答案'] = clean_df['答案'].astype(str).str.strip()

# Save as the final training dataset
clean_df.to_csv("dataset/HW2_cleaned.csv", index=False)

print(f"Cleaning complete！")
print(f"Original row count: {len(df)}")
print(f"Cleaned row count: {len(clean_df)}")
print("Distribution：")
print(clean_df['答案'].value_counts())
print("\n dataset/HW2_cleaned.csv is ready for training")