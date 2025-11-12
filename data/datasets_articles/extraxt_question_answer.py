import json

# Load the input data
with open("CyberMetric-2000-v1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extract question and correct answer pairs
qa_pairs = []
for item in data["questions"]:
    question = item["question"]
    correct_answer_key = item["solution"]
    correct_answer_text = item["answers"][correct_answer_key]
    qa_pairs.append({
        "question": question,
        "answer": correct_answer_text
    })

# Save the extracted pairs into a new JSON file
with open("CyberMetric-2000-qa.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, indent=4, ensure_ascii=False)

print("✅ Extracted questions and answers saved to 'qa_pairs.json'")
