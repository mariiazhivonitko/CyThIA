import pandas as pd
import json

# ---- Configuration ----
input_file = "CyberMetric-2000-qa.json"      
output_file = "CyberMetric-2000-qa.xlsx"     

# ---- Load JSON ----
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- Convert to DataFrame ----
# If your JSON is a list of dictionaries, e.g. [{"question": "...", "answer": "..."}]
df = pd.DataFrame(data)

# ---- Save to Excel ----
df.to_excel(output_file, index=False)

print(f"Successfully saved {len(df)} rows to '{output_file}'")
