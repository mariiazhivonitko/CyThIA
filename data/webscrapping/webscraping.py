from docx import Document
from bs4 import BeautifulSoup
import requests
import pandas as pd

doc = Document("question_answer_urls.docx")

#Extract urls from document
urls = []
for para in doc.paragraphs:
    if para.text.startswith("http"):
        urls.append(para.text)

#Let's extract the first url and scrape it
url = urls[-1]

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36"
}

response = requests.get(url, headers=headers)
response.raise_for_status()
soup = BeautifulSoup(response.content, "html.parser")

print(url)
print(soup.title.string)

question_answer_pairs = []

""" for h3 in soup.find_all("h3"): 
    question = h3.get_text(strip=True)

    answer_parts = [] 
    for sibling in h3.find_next_siblings(): 
        if sibling.name and sibling.name.startswith("h3"):
            break
        if sibling.name == "p":
            answer_parts.append(sibling.get_text(strip=True))   

    if answer_parts:
        answer = " ".join(answer_parts).strip()
        question_answer_pairs.append((question, answer)) """

# for url = urls[2] :"https://www.simplilearn.com/tutorials/cyber-security-tutorial/cyber-security-interview-questions"
""" 
for h3 in soup.find_all("h3"):
    question = h3.get_text(strip=True)

    # Collect answer parts: paragraphs and list items after <h3>
    answer_parts = []
    for sibling in h3.find_next_siblings():
        # Stop when we hit another question <h3>
        if sibling.name == "h3":
            break

        if sibling.name == "p":
            answer_parts.append(sibling.get_text(strip=True))

        elif sibling.name == "ul":
            for li in sibling.find_all("li"):
                answer_parts.append("• " + li.get_text(strip=True))
 


    if answer_parts:
        answer = "\n".join(answer_parts)
        question_answer_pairs.append((question, answer))

for question, answer in question_answer_pairs:
    print("Q:", question)
    print("A:", answer)
"""

# for url = urls[4] :"https://cybertalents.com/blog/41-questions-to-help-you-prepare-for-a-cybersecurity-interview"
# Loop through all h3 > span (questions)


for h3 in soup.find_all("h3"):
    # Extract question text (handles span, b, strong, etc.)
    question = h3.get_text(" ", strip=True)
    if not question:
        continue

    answer_parts = []
    for sibling in h3.find_next_siblings():
        if sibling.name == "h3":  # stop at next question
            break

        if sibling.name in ["p", "div", "b", "strong"]:
            text = sibling.get_text(" ", strip=True)
            if text:
                answer_parts.append(text)

        elif sibling.name in ["ul", "ol"]:
            for li in sibling.find_all("li"):
                text = li.get_text(" ", strip=True)
                if text:
                    answer_parts.append("• " + text)

    if answer_parts:
        answer = " ".join(answer_parts).strip()
        question_answer_pairs.append((question, answer))

# Convert to DataFrame
df = pd.DataFrame(question_answer_pairs, columns=["Question", "Answer"])
df.to_excel("cybersecurity_qa.xlsx", index=False)

print(f"Extracted {len(df)} Q&A pairs and saved to cybersecurity_qa.xlsx")
