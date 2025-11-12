import json
import random

# ----------------------
# CONFIG
# ----------------------
input_file = "./cybersecurity_messages.jsonl" 
train_file = "./train.jsonl"
val_file = "./validation.jsonl"
test_file = "./test.jsonl"

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1
seed = 42

# ----------------------
# LOAD DATA
# ----------------------
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

print(f"Total examples: {len(data)}")

# Shuffle data for random split
random.seed(seed)
random.shuffle(data)

# ----------------------
# SPLIT DATA
# ----------------------
n_total = len(data)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
n_test = n_total - n_train - n_val

train_data = data[:n_train]
val_data = data[n_train:n_train + n_val]
test_data = data[n_train + n_val:]

print(f"Train: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

# ----------------------
# SAVE SPLITS
# ----------------------
def save_jsonl(filename, dataset):
    with open(filename, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

save_jsonl(train_file, train_data)
save_jsonl(val_file, val_data)
save_jsonl(test_file, test_data)

print("Datasets saved!")
