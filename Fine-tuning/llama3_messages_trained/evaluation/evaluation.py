import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import csv

# ----------------------
# CONFIG
# ----------------------
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
FINETUNED_MODEL = "mariiazhiv/CyThIA-llama3.1-8B-messages"
HF_DATASET = "mariiazhiv/Cybersecurity_messages"
TEST_SPLIT = "validation"
MAX_TOKENS = 280
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_CSV = "evaluation_chat_results.csv"

# ----------------------
# Load tokenizer
# ----------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|pad|>"

# ----------------------
# Load base model
# ----------------------
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
)

# ----------------------
# Resize embeddings
# ----------------------
if model.get_input_embeddings().num_embeddings != len(tokenizer):
    print(f"Resizing embeddings → {len(tokenizer)} tokens")
    model.resize_token_embeddings(len(tokenizer))

# ----------------------
# Load LoRA adapter
# ----------------------
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, FINETUNED_MODEL)
model.eval()

# ----------------------
# Load HF dataset
# ----------------------
dataset = load_dataset(HF_DATASET, split=TEST_SPLIT)
print(f"Loaded {len(dataset)} examples from {HF_DATASET}:{TEST_SPLIT}")

# ----------------------
# Embedding model for semantic similarity
# ----------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

smooth_fn = SmoothingFunction().method1
rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
results = []


# ----------------------
# Evaluation loop
# ----------------------
for i, item in enumerate(dataset):

    messages = item["messages"]

    # Get reference assistant message
    reference = messages[-1]["content"]

    prompt = tokenizer.apply_chat_template(
        messages[:-1],   # exclude last assistant message
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )

    # Decode
    generated_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # 1️ Semantic similarity
    emb_pred = embedder.encode(prediction, convert_to_tensor=True)
    emb_true = embedder.encode(reference, convert_to_tensor=True)
    sem_sim = util.pytorch_cos_sim(emb_pred, emb_true).item()

    # 2️ BLEU
    reference_tokens = [reference.split()]
    prediction_tokens = prediction.split()
    bleu = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smooth_fn)

    # 3️ BERTScore
    P, R, F1 = bert_score([prediction], [reference], lang="en", rescale_with_baseline=True)
    bert_f1 = F1.item()

    # 4️ ROUGE scores
    rouge_scores = rouge.score(reference, prediction)
    rouge1_f = rouge_scores["rouge1"].fmeasure
    rouge2_f = rouge_scores["rouge2"].fmeasure
    rougeL_f = rouge_scores["rougeL"].fmeasure

    results.append({
        "prediction": prediction,
        "reference": reference,
        "semantic_similarity": sem_sim,
        "bleu": bleu,
        "bert_score": bert_f1,
        "rouge1": rouge1_f,
        "rouge2": rouge2_f,
        "rougeL": rougeL_f
    })

    print(
        f"[{i+1}/{len(dataset)}] "
        f"SemSim={sem_sim:.4f}, BLEU={bleu:.4f}, "
        f"BERT={bert_f1:.4f}, "
        f"R1={rouge1_f:.4f}, R2={rouge2_f:.4f}, RL={rougeL_f:.4f}"
    )

# ----------------------
# Save CSV
# ----------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    fieldnames = [
        "prediction", "reference",
        "semantic_similarity",
        "bleu", "bert_score",
        "rouge1", "rouge2", "rougeL"
    ]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

# ----------------------
# Compute averages
# ----------------------
avg_sem   = sum(r["semantic_similarity"] for r in results) / len(results)
avg_bleu  = sum(r["bleu"] for r in results) / len(results)
avg_bert  = sum(r["bert_score"] for r in results) / len(results)
avg_r1    = sum(r["rouge1"] for r in results) / len(results)
avg_r2    = sum(r["rouge2"] for r in results) / len(results)
avg_rl    = sum(r["rougeL"] for r in results) / len(results)

print("\n==== AVERAGE METRICS ====")
print(f"Semantic similarity: {avg_sem:.4f}")
print(f"BLEU:                {avg_bleu:.4f}")
print(f"BERTScore F1:        {avg_bert:.4f}")
print(f"ROUGE-1 F:           {avg_r1:.4f}")
print(f"ROUGE-2 F:           {avg_r2:.4f}")
print(f"ROUGE-L F:           {avg_rl:.4f}")

with open("average_chat_scores.txt", "w", encoding="utf-8") as f:
    f.write("==== AVERAGE METRICS ====\n")
    f.write(f"Semantic similarity: {avg_sem:.4f}\n")
    f.write(f"BLEU: {avg_bleu:.4f}\n")
    f.write(f"BERTScore F1: {avg_bert:.4f}\n")
    f.write(f"ROUGE-1: {avg_r1:.4f}\n")
    f.write(f"ROUGE-2: {avg_r2:.4f}\n")
    f.write(f"ROUGE-L: {avg_rl:.4f}\n")
