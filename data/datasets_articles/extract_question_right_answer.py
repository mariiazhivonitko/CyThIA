import json
import pandas as pd

# Path to your JSON file
input_file = "MCQs_2730_english.jsonl"
output_file = "MCQs_2730_english_qa_pairs.xlsx"

# Mapping from letter labels to indices
label_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}

qa_pairs = []

# --- For JSON with one object per line ---
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():  # skip empty lines
            obj = json.loads(line)
            label = obj.get("label")
            if label not in label_to_index:
                continue
            correct_index = label_to_index[label]
            question = obj.get("question", "")
            correct_answer = obj.get("answers", [])[correct_index]
            qa_pairs.append({"Question": question, "Answer": correct_answer})

# --- For JSON as a list (uncomment if needed) ---
# with open(input_file, "r", encoding="utf-8") as f:
#     data = json.load(f)
#     for obj in data:
#         label = obj.get("label")
#         if label not in label_to_index:
#             continue
#         correct_index = label_to_index[label]
#         question = obj.get("question", "")
#         correct_answer = obj.get("answers", [])[correct_index]
#         qa_pairs.append({"Question": question, "Answer": correct_answer})

# Convert to pandas DataFrame
df = pd.DataFrame(qa_pairs)

# Save to Excel
df.to_excel(output_file, index=False)

print(f"Extracted {len(qa_pairs)} question - answer pairs saved to {output_file}")
