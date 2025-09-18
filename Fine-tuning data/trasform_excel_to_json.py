import pandas as pd
import json
import re

def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return text
    # Remove leading/trailing spaces
    text = text.strip()
    # Collapse multiple spaces inside
    text = " ".join(text.split())
    # Normalize curly quotes
    text = re.sub("[“”]", '"', text)  # curly double → straight double
    text = re.sub("[‘’]", "'", text)  # curly single → straight single
    return text

# Load Excel file
df = pd.read_excel("question_answer_pairs_cybersecurity.xlsx")

# Apply cleaning to each cell
df = df.applymap(normalize_text)

# Convert to Alpaca-style JSONL
with open("cybersecurity_qa.jsonl", "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        record = {
            "instruction": row["Question"],
            "input": "",
            "output": row["Answer"]
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print("✅ Cleaned dataset saved as cybersecurity_qa.jsonl")
