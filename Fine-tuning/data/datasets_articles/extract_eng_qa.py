import json

input_file = "data.jsonl"      # your input file
output_file = "english.jsonl"  # filtered output file

with open(input_file, "r", encoding="utf-8") as infile, \
     open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        if not line.strip():
            continue  # skip empty lines
        obj = json.loads(line)
        if obj.get("language") == "English":
            outfile.write(json.dumps(obj, ensure_ascii=False) + "\n")

print("Filtered English data written to", output_file)