import json
import requests
import trafilatura
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# Load original scraped documents
input_path = Path('../data/cleaned_docs/documents.jsonl')
output_path = Path('../data/cleaned_docs/enriched_documents.jsonl')

# Common subpages to check, this list can be extended
subpages = ['about-us', 'products', 'services']

# Load existing homepage documents
with open(input_path, 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f]

# Function to fetch and extract readable content
def fetch_content(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded)
    except Exception as e:
        print(f"Error fetching {url}: {e}")
    return None

# Enrich each company doc with subpage content
enriched_docs = []

for doc in tqdm(docs):
    base_url = doc['url'].rstrip('/')
    merged_content = doc['content']
    visited_subpages = []

    for sub in subpages:
        sub_url = f"{base_url}/{sub}"
        content = fetch_content(sub_url)
        if content:
            merged_content += f"\n\n### {sub.upper()} PAGE CONTENT ###\n" + content
            visited_subpages.append(sub)

    enriched_doc = {
        'company_name': doc['company_name'],
        'base_url': doc['url'],
        'source_pages': ['home'] + visited_subpages,
        'content': merged_content,
        'timestamp': datetime.utcnow().isoformat()
    }
    enriched_docs.append(enriched_doc)

# Save enriched output
with open(output_path, 'w', encoding='utf-8') as f:
    for doc in enriched_docs:
        f.write(json.dumps(doc) + '\n')

print("✅ Enrichment complete. Saved to:", output_path)
