from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import re
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

with open("./download/MSFT/MSFT-20250730.htm", "r", encoding="utf-8", errors="ignore") as f:
    html_text = f.read()
soup = BeautifulSoup(html_text, "lxml")
candidates = []
for tag in soup.find_all():
    txt = tag.get_text(" ", strip=True).upper()
    if "INDEX" in txt:
        anchors = tag.find_all("a", href=re.compile(r"^#"))
        if len(anchors) >= 0:
            candidates.append((tag, len(anchors)))
candidates.sort(key=lambda x: x[1], reverse=True)
toc_tag = candidates[0][0]
anchor_texts = [a.get_text(" ", strip=True) for a in toc_tag.find_all("a", href=re.compile(r"^#"))]

unique_list = list(dict.fromkeys(anchor_texts))
noisy_sentences=['Table of Contents']

cleaned_list = []
for item in unique_list:
    if item == "Form 10-K Summary": 
        break
    if not item.isdigit() and item not in noisy_sentences:
        cleaned_list.append(item)
        print (item)

print(cleaned_list)
print(len(cleaned_list))
