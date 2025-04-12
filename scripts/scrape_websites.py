import pandas as pd
import requests
import trafilatura
import json
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

# --- Config ---
INPUT_CSV = '../data/websites.csv'
OUTPUT_PATH = Path('../data/cleaned_docs/documents.jsonl')
MAX_COMPANIES = 400  # Set to None to scrape all; or e.g., 50 for testing

# --- Load CSV ---
df = pd.read_csv(INPUT_CSV)
df = df[['CompanyName', 'Website']].dropna()

# Optional: Limit number of companies
if MAX_COMPANIES:
    df = df.sample(n=MAX_COMPANIES, random_state=42)

print(f"🔍 Scraping {len(df)} company websites...")

# --- Scraping logic ---
def extract_text(url):
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            return trafilatura.extract(downloaded)
    except Exception as e:
        print(f"⚠️ Error fetching {url}: {e}")
    return None

# --- Write results ---
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for _, row in tqdm(df.iterrows(), total=len(df)):
        content = extract_text(row['Website'])
        if content:
            doc = {
                'company_name': row['CompanyName'],
                'url': row['Website'],
                'content': content,
                'timestamp': datetime.utcnow().isoformat()
            }
            f.write(json.dumps(doc) + '\n')

print(f"✅ Done. Scraped content saved to {OUTPUT_PATH}")
