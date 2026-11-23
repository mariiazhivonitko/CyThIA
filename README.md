# CyThIA – Conversational Cyber Threat Intelligence Advisor

**CyThIA** is a research prototype of a conversational AI assistant that answers cybersecurity and cyber threat intelligence (CTI) questions. It was built for my bachelor's thesis *"Designing a Conversational Cyber Threat Advisor Using Domain-Tuned Conversational AI"* (Oulu University of Applied Sciences, in collaboration with the Future Computing Group, University of Oulu, 2026).

This repository covers the **model side** of the project: **dataset creation → LoRA fine-tuning of three open LLMs → evaluation with automatic metrics → selection of the foundation model for the chatbot.**

The chatbot itself (implementation, deployment, and evaluation with DeepEval using an LLM as a judge) is in a separate repository: **[chatbot_on_runpod](https://github.com/mariiazhivonitko/chatbot_on_runpod)**.

> ⚠️ CyThIA is an exploratory prototype for research and learning. It is **not** intended for real-world security decisions or autonomous operations.

- 📄 **Thesis (open access):** [Designing a Conversational Cyber Threat Advisor Using Domain-Tuned Conversational AI](https://urn.fi/URN:NBN:fi:amk-202604015472)
- 🔗 **Chatbot repo:** [chatbot_on_runpod](https://github.com/mariiazhivonitko/chatbot_on_runpod) (Streamlit + FastAPI on RunPod, DeepEval evaluation)
- 🤗 **Datasets on Hugging Face:** [`mariiazhiv/cybersecurity_qa`](https://huggingface.co/datasets/mariiazhiv/cybersecurity_qa) · [`mariiazhiv/Cybersecurity_messages`](https://huggingface.co/datasets/mariiazhiv/Cybersecurity_messages)

---

## Overview

<!-- Add a simple architecture diagram here, e.g. images/architecture.png -->

```
 Web scraping + public CTI datasets
              │
              ▼
   Cleaned Q&A pairs  ──►  train / validation / test split
              │
              ▼
   LoRA / PEFT fine-tuning with Axolotl (RunPod A40 GPU)
   Mistral-7B-Instruct · Llama-3.1-8B-Instruct · Llama-2-13B-chat
              │
              ▼
   Evaluation: semantic similarity, BERTScore, BLEU, ROUGE
              │
              ▼
   Best model: Mistral-7B-Instruct  ──►  CyThIA-Mistral
                                         (used in chatbot_on_runpod)
```

## Tech stack

Python · Hugging Face Transformers & Datasets · PEFT / LoRA · Axolotl · RunPod (GPU) · Sentence-Transformers · BERTScore · BLEU / ROUGE · Ollama (GraphRAG experiment)

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
│   ├── evaluation.py         # evaluation script (semantic similarity, BERTScore, BLEU, ROUGE)
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
- Three open-source, lightweight dialog-generation models were fine-tuned: **Mistral-7B-Instruct**, **Llama-3.1-8B-Instruct** and **Llama-2-13B-chat**.
- **Mistral-7B-Instruct** was selected as the foundation model for the chatbot (**CyThIA-Mistral**).

## Evaluation & model selection

Evaluating open-ended generated text is hard, and no single automatic metric is enough. So the three fine-tuned models were compared on the held-out **test split** with several complementary metrics, covering lexical overlap, semantic alignment and content coverage. The goal was to choose the best **foundation model for the chatbot**.

**Metrics**

| Metric | What it measures |
|---|---|
| Semantic similarity | Cosine similarity between model answer and reference answer embeddings (`all-MiniLM-L6-v2`) |
| BERTScore F1 | Token-level semantic overlap using contextual embeddings (rescaled with baseline) |
| BLEU | Exact n-gram overlap with the reference |
| ROUGE-1 / ROUGE-2 / ROUGE-L | Unigram, bigram and longest-common-subsequence overlap with the reference |

**Fine-tuning evaluation scores** (best value per metric in bold)

| Model | Semantic sim. | BERTScore F1 | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|---|---|
| **Mistral-7B-Instruct** | **0.7823** | **0.2008** | **0.0370** | **0.3131** | **0.1049** | **0.2041** |
| Llama-3.1-8B-Instruct | 0.7533 | 0.1958 | 0.0323 | 0.3093 | 0.1016 | 0.1993 |
| Llama-2-13B-chat | 0.7543 | 0.1810 | 0.0330 | 0.3071 | 0.1005 | 0.2030 |

**Mistral-7B-Instruct consistently outperformed both Llama models on all six metrics.** Its higher semantic similarity and BERTScore show better semantic alignment with the reference answers, and its higher ROUGE scores show better content coverage. BLEU is low for all models, which is expected for open-ended conversational text, but Mistral still scored highest. Based on these results, Mistral-7B-Instruct was selected as the foundation model for the chatbot (**CyThIA-Mistral**).

**Next step:** CyThIA-Mistral became the model behind the chatbot. Its evaluation as a complete conversational system (DeepEval, LLM-as-a-judge) is documented in **[chatbot_on_runpod](https://github.com/mariiazhivonitko/chatbot_on_runpod)**.

Per-example results are in `Fine-tuning/*/evaluation_results*.csv` and averages in `average_scores*.txt`.

## How to run

```bash
# 1. Clone
git clone https://github.com/mariiazhivonitko/CyThIA.git
cd CyThIA

# 2. Install dependencies
pip install -r requirements.txt

# 3. Fine-tune with Axolotl (GPU required)
accelerate launch -m axolotl.cli.train Fine-tuning/config.yaml

# 4. Evaluate a fine-tuned model on the test split
python Fine-tuning/evaluation.py
```

## Experiments & next steps

- **GraphRAG:** I explored Microsoft GraphRAG with locally hosted models (Ollama: phi3 + nomic-embed-text) to add retrieval over CTI reports. The integration did not reach working quality within the thesis timeframe, so it is not part of the final system.
- **Other training runs:** I also tried training Llama-3.1-8B on a chat "messages" version of the dataset, and tested Mixtral and Flan-T5-XXL. These preliminary runs are in `Fine-tuning/` but are not part of the thesis comparison.
- **Next steps:** a larger and higher-quality dataset, retrieval-augmented generation (RAG) over up-to-date threat reports, and stronger safety guardrails.

## Acknowledgements

This project was supervised by **Prasasthy K B**, Future Computing Group, University of Oulu.

## Author

**Mariia Zhivonitko** – Data & AI | LLM fine-tuning, evaluation, RAG | Oulu, Finland

[LinkedIn](https://www.linkedin.com/in/mariia-zhivonitko/)
