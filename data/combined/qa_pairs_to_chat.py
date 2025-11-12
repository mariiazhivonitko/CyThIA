import json

with open("cybersecurity_qa_united.jsonl", "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

with open("cybersecurity_messages.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        formatted = {
            "messages": [
                {"role": "system", "content": "You are CyThIA, a helpful cybersecurity assistant."},
                {"role": "user", "content": item["question"]},
                {"role": "assistant", "content": item["answer"]}
            ]
        }
        f.write(json.dumps(formatted) + "\n")
