import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from bert_score import score as bert_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import csv

# ----------------------
# CONFIG
# ----------------------
BASE_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
FINETUNED_MODEL = "mariiazhiv/CyTHIA-Mixtral-8x7B"  # folder with LoRA adapters and tokenizer
HF_DATASET = "mariiazhiv/cybersecurity_qa"  # Hugging Face dataset repo
TEST_SPLIT = "test"
MAX_TOKENS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_CSV = "evaluation_results.csv"

# ----------------------
# LOAD TOKENIZER
# ----------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = "<|pad|>"

# ----------------------
# LOAD BASE MODEL
# ----------------------
print("Loading base model on CPU (safe mode)...")
# load on CPU to avoid meta tensors
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map={"": "cpu"},                # ensure tensors are actual CPU tensors
    torch_dtype=torch.float32,            # load in fp32 on CPU for safe init
    low_cpu_mem_usage=True               # avoid lazy/meta init
)
# ----------------------
# RESIZE TOKEN EMBEDDINGS TO MATCH TOKENIZER
# ----------------------
if model.get_input_embeddings().num_embeddings != len(tokenizer):
    print(f"Resizing model embeddings from {model.get_input_embeddings().num_embeddings} to {len(tokenizer)}")
    model.resize_token_embeddings(len(tokenizer))

# ----------------------
# LOAD LoRA ADAPTER
# ----------------------
print("Loading LoRA adapters...")
model = PeftModel.from_pretrained(model, FINETUNED_MODEL)
model.eval()

# now move/cast model to target device
if DEVICE == "cuda":
    # move to GPU and cast to float16 (fp16). If you prefer bfloat16 and your GPU supports it, change here.
    model = model.to(torch.bfloat16).to(DEVICE)

# ----------------------
# LOAD TEST DATASET FROM HF
# ----------------------
dataset = load_dataset(HF_DATASET, split=TEST_SPLIT)
print(f"Loaded {len(dataset)} test examples from {HF_DATASET}, split={TEST_SPLIT}")

# ----------------------
# LOAD EMBEDDING MODEL
# ----------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ----------------------
# LOAD ROUGE SCORER
# ----------------------
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

# ----------------------
# EVALUATION
# ----------------------
results = []
smooth_fn = SmoothingFunction().method1

for i, item in enumerate(dataset):
    # Prepare prompt
    prompt = f"{item['instruction']}\n{item.get('input', '')}".strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # Generate model output
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=MAX_TOKENS)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

    reference = item["output"]

    # 1️⃣ Semantic similarity (cosine)
    emb_pred = embedder.encode(prediction, convert_to_tensor=True)
    emb_true = embedder.encode(reference, convert_to_tensor=True)
    sem_sim = util.pytorch_cos_sim(emb_pred, emb_true).item()

    # 2️⃣ BLEU score
    reference_tokens = [reference.split()]
    prediction_tokens = prediction.split()
    bleu = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smooth_fn)

    # 3️⃣ BERTScore
    P, R, F1 = bert_score([prediction], [reference], lang="en", rescale_with_baseline=True)
    bert_f1 = F1.item()

    # 4️⃣ ROUGE scores
    rouge_scores = scorer.score(reference, prediction)
    rouge1_f = rouge_scores["rouge1"].fmeasure
    rouge2_f = rouge_scores["rouge2"].fmeasure
    rougeL_f = rouge_scores["rougeL"].fmeasure
 

    results.append({
        "instruction": item["instruction"],
        "prediction": prediction,
        "reference": reference,
        "semantic_similarity": sem_sim,
        "bleu": bleu,
        "bert_score": bert_f1,
                "rouge1": rouge1_f,
        "rouge2": rouge2_f,
        "rougeL": rougeL_f
    })

    print(f"Example {i+1}: SemSim={sem_sim:.4f}, BLEU={bleu:.4f}, BERTScore={bert_f1:.4f}", f"ROUGE-2={rouge2_f:.4f}, ROUGE-L={rougeL_f:.4f}")

# ----------------------
# SAVE RESULTS TO CSV
# ----------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["instruction", "prediction", "reference", "semantic_similarity", "bleu", "bert_score", "rouge1", "rouge2", "rougeL"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# ----------------------
# AVERAGE METRICS
# ----------------------
avg_sem = sum(r["semantic_similarity"] for r in results)/len(results)
avg_bleu = sum(r["bleu"] for r in results)/len(results)
avg_bert = sum(r["bert_score"] for r in results)/len(results)
avg_rouge1 = sum(r["rouge1"] for r in results)/len(results)
avg_rouge2 = sum(r["rouge2"] for r in results)/len(results)
avg_rougeL = sum(r["rougeL"] for r in results)/len(results)


print("\n==== AVERAGE SCORES ====")
print(f"Semantic similarity: {avg_sem:.4f}")
print(f"BLEU score: {avg_bleu:.4f}")
print(f"BERTScore F1: {avg_bert:.4f}")
print(f"ROUGE-1 F1: {avg_rouge1:.4f}")
print(f"ROUGE-2 F1: {avg_rouge2:.4f}")
print(f"ROUGE-L F1: {avg_rougeL:.4f}")

# ----------------------
# SAVE AVERAGE SCORES TO FILE
# ----------------------
with open("average_scores.txt", "w", encoding="utf-8") as f:
    f.write("==== AVERAGE SCORES ====\n")
    f.write(f"Semantic similarity: {avg_sem:.4f}\n")
    f.write(f"BLEU score: {avg_bleu:.4f}\n")
    f.write(f"BERTScore F1: {avg_bert:.4f}\n")
    f.write(f"ROUGE-1 F1: {avg_rouge1:.4f}\n")
    f.write(f"ROUGE-2 F1: {avg_rouge2:.4f}\n")
    f.write(f"ROUGE-L F1: {avg_rougeL:.4f}\n")
 
