import json
from asyncio import timeout

import requests
import trafilatura
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# --- Config ---
INPUT_PATH = Path('../data/cleaned_docs/documents.jsonl')
OUTPUT_PATH = Path('../data/cleaned_docs/enriched_documents.jsonl')
SUBPAGES = ['about', 'about-us', 'products', 'services']

# --- Load scraped homepage documents ---
with open(INPUT_PATH, 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f]

# --- Helper to fetch and extract readable text from a subpage ---
def fetch_content(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        if response.status_code == 200:
            return trafilatura.extract(response.text)
        else:
            print(f"⚠️ Non-200 status for {url}: {response.status_code}")
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
    return None

# --- Enrich content ---
enriched_docs = []

for doc in tqdm(docs):
    base_url = doc['url'].rstrip('/')
    merged_content = doc['content']
    visited_subpages = []

    for sub in SUBPAGES:
        sub_url = f"{base_url}/{sub}"
        content = fetch_content(sub_url)
        if content:
            merged_content += f"\n\n### {sub.upper()} PAGE CONTENT ###\n{content}"
            visited_subpages.append(sub)

    enriched_doc = {
        'company_name': doc['company_name'],
        'base_url': doc['url'],
        'source_pages': ['home'] + visited_subpages,
        'content': merged_content,
        'timestamp': datetime.utcnow().isoformat()
    }
    enriched_docs.append(enriched_doc)

# --- Save enriched documents ---
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for doc in enriched_docs:
        f.write(json.dumps(doc) + '\n')

print(f"✅ Enrichment complete. Saved to: {OUTPUT_PATH}")
