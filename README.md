# CyThIA – Conversational Cyber Threat Intelligence Advisor

**CyThIA** is a research prototype of a conversational AI assistant that answers cybersecurity and cyber threat intelligence (CTI) questions. It was built for my bachelor's thesis *"Designing a Conversational Cyber Threat Advisor Using Domain-Tuned Conversational AI"* (Oulu University of Applied Sciences, in collaboration with the Future Computing Group, University of Oulu, 2026).

The project covers the full pipeline: **dataset creation → LoRA fine-tuning of several open LLMs → automated evaluation (LLM-as-judge) → deployment as a chat app.**

> ⚠️ CyThIA is an exploratory prototype for research and learning. It is **not** intended for real-world security decisions or autonomous operations.

🔗 **Chat app:** [chatbot_on_runpod](https://github.com/mariiazhivonitko/chatbot_on_runpod) (Streamlit frontend + FastAPI backend on RunPod)
🤗 **Datasets on Hugging Face:** [`mariiazhiv/cybersecurity_qa`](https://huggingface.co/datasets/mariiazhiv/cybersecurity_qa) · [`mariiazhiv/Cybersecurity_messages`](https://huggingface.co/datasets/mariiazhiv/Cybersecurity_messages)

---

## Overview

<!-- Add a simple architecture diagram here, e.g. images/architecture.png -->

```
 Web scraping + public CTI datasets
              │
              ▼
   Cleaned Q&A pairs  ──►  train / validation / test split
              │                (Q&A and chat "messages" formats)
              ▼
   LoRA / PEFT fine-tuning with Axolotl (RunPod A40 GPU)
   Mistral-7B · Llama 3 · Llama 2-13B-chat · (Mixtral, Flan-T5 tested)
              │
              ▼
   Evaluation with DeepEval (GPT-4.1 as judge)
              │
              ▼
   CyThIA-Mistral served via FastAPI  ──►  Streamlit chat UI
```

## Tech stack

Python · Hugging Face Transformers · PEFT / LoRA · Axolotl · RunPod (GPU) · CSC cPouta · DeepEval · FastAPI · Streamlit · Ollama

## Repository structure

```
CyThIA/
├── data/
│   ├── webscrapping/         # scraping script + scraped cybersecurity Q&A pairs
│   ├── datasets_articles/    # extraction scripts for public research datasets (CyberMetric, MCQs, SAQs)
│   └── combined/             # final merged dataset, train/val/test splits,
│                             # Q&A → chat-format conversion
├── Fine-tuning/
│   ├── config.yaml           # Axolotl LoRA configuration
│   ├── evaluation.py         # DeepEval evaluation script
│   ├── mistral/              # evaluation results for the selected model
│   ├── llama3/, llama3_messages_trained/, llama2-13B-chat/
│   └── mixtral/, flan_t5_xxl/   # earlier experiments
└── graphrag/                 # experimental GraphRAG indexing (see "Experiments")
```

## Dataset

- Cybersecurity Q&A pairs collected by **web scraping** and extracted from **public research datasets** (CyberMetric-2000, MCQ/SAQ cybersecurity question sets).
- Cleaned and merged into **502 curated Q&A pairs**, then split into train / validation / test sets.
- Provided in two formats: plain **question–answer pairs** and **chat "messages"** for instruction-tuned models.
- Published on Hugging Face (links above).

## Fine-tuning

- Method: **LoRA / PEFT** with **Axolotl** on a RunPod **A40** GPU.
- Models compared: **Mistral-7B-v0.3**, **Llama 3.1-8B-Instruct**, **Llama 2-13B-chat**. Mixtral and Flan-T5-XXL were also tried.
- **Mistral-7B** was selected as the final base model (**CyThIA-Mistral**).

## Evaluation

The models were evaluated on the held-out test set with **DeepEval**, using **GPT-4.1 as an LLM judge**.

| Metric | CyThIA-Mistral |
|---|---|
| Answer relevancy | ~0.79 |
| Correctness | ~0.35 |
| Conciseness | ~0.32 |
| Coherence | ~0.25 |

**Interpretation:** answers are generally on-topic, but correctness and coherence are too low for real-world use. This is why CyThIA is presented as an exploratory prototype. The results point to dataset size and quality as the main areas for improvement.

Per-model results are in `Fine-tuning/*/average_scores*.txt` and `evaluation_results*.csv`.

## How to run

```bash
# 1. Clone
git clone https://github.com/mariiazhivonitko/CyThIA.git
cd CyThIA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fine-tune with Axolotl (GPU required)
accelerate launch -m axolotl.cli.train Fine-tuning/config.yaml

# 4. Evaluate (needs an OpenAI API key for the GPT-4.1 judge)
export OPENAI_API_KEY=...
python Fine-tuning/evaluation.py
```

## Experiments & next steps

- **GraphRAG:** I explored Microsoft GraphRAG with locally hosted models (Ollama: phi3 + nomic-embed-text) to add retrieval over CTI reports. The integration did not reach working quality within the thesis timeframe, so it is not part of the final system.
- **Next steps:** a larger and higher-quality dataset, retrieval-augmented generation (RAG) over up-to-date threat reports, and stronger safety guardrails.

## Author

**Mariia Zhivonitko** – Data & AI | LLM fine-tuning, evaluation, RAG | Oulu, Finland
<!-- Add your LinkedIn link here -->
