import json
import os
import re
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File paths
input_path = Path('../data/cleaned_docs/enriched_documents.jsonl')
output_path = Path('../data/cleaned_docs/enriched_with_metadata.jsonl')

# Load enriched documents
with open(input_path, 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f]

# Prompt template for GPT
def build_prompt(content):
    return f"""
Extract the following information from the company description below:
1. What is the pricing model? If specific prices are mentioned, estimate them.
2. Does the company sell to businesses (B2B), consumers (B2C), or both?
3. Is there any product pricing mentioned? If yes, include it.

Respond in JSON format with these fields:
- business_model
- price_hint
- pricing_mentioned

Company Description:
{content[:3000]}
"""

# Process and enrich each document
enriched_output = []

for doc in tqdm(docs):
    prompt = build_prompt(doc['content'])
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a data extraction assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw_response = response.choices[0].message.content.strip()

        # Try to extract valid JSON block
        match = re.search(r'\{.*?\}', raw_response, re.DOTALL)
        if match:
            try:
                metadata = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"⚠️ JSON decode error for {doc['company_name']}:")
                print(raw_response)
                continue
        else:
            print(f"⚠️ No JSON found in response for {doc['company_name']}:")
            print(raw_response)
            continue

        # Merge result with original doc
        enriched_doc = {
            **doc,
            "metadata_extracted": metadata,
            "llm_model": "gpt-4o",
            "processed_at": datetime.utcnow().isoformat()
        }
        enriched_output.append(enriched_doc)

    except Exception as e:
        print(f"⚠️ Error processing {doc['company_name']}: {e}")

# Save the results
with open(output_path, 'w', encoding='utf-8') as f:
    for doc in enriched_output:
        f.write(json.dumps(doc) + '\n')

print("✅ Metadata extraction complete. Saved to:", output_path)
