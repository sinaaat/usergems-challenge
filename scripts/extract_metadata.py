import json
import os
import re
from tqdm import tqdm
from openai import OpenAI
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=api_key)

# File paths
input_path = Path('../data/cleaned_docs/enriched_documents.jsonl')
output_path = Path('../data/cleaned_docs/enriched_with_metadata.jsonl')

# Load enriched documents
with open(input_path, 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f]

# Prompt template builder
def build_prompt(content):
    return f"""
    You are a data extraction assistant. From the company description below, extract:
    
    1. The business model (B2B, B2C, or both)
    2. Pricing info (if available or implied)
    3. Whether any pricing is mentioned at all
    
    Please infer from vague hints like "subscription", "contact us for pricing", "enterprise plan", etc.
    
    Examples:
    - "Enterprise plan available" → pricing_mentioned: true, price_hint: "enterprise-level pricing"
    - "Starts from $99/month" → pricing_mentioned: true, price_hint: "$99/month"
    - "Contact us for pricing" → pricing_mentioned: true, price_hint: "pricing upon request"
    - No mention of pricing → pricing_mentioned: false, price_hint: null
    
    Respond in valid JSON like this:
    {{
      "business_model": "B2B",
      "price_hint": "subscription model, estimated under $1000",
      "pricing_mentioned": true
    }}
    
    Company Description:
    {content[:3000]}
    """

# Process and enrich each document
enriched_output = []

for idx, doc in enumerate(tqdm(docs)):
    prompt = build_prompt(doc['content'])

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw_response = response.choices[0].message.content.strip()

        # Extract JSON block from response
        match = re.search(r'\{.*?\}', raw_response, re.DOTALL)
        if match:
            try:
                metadata = json.loads(match.group(0))
            except json.JSONDecodeError:
                print(f"⚠️ JSON decode error for {doc['company_name']}:")
                print(raw_response)
                continue
        else:
            print(f"⚠️ No JSON found for {doc['company_name']}")
            print(raw_response)
            continue

        enriched_doc = {
            **doc,
            "metadata_extracted": metadata,
            "llm_model": "gpt-4o",
            "processed_at": datetime.utcnow().isoformat()
        }
        enriched_output.append(enriched_doc)

        if (idx + 1) % 10 == 0:
            print(f"✅ Processed {idx + 1} companies...")

    except Exception as e:
        print(f"⚠️ Error with {doc['company_name']}: {e}")
        continue

# Save output
with open(output_path, 'w', encoding='utf-8') as f:
    for doc in enriched_output:
        f.write(json.dumps(doc) + '\n')

print(f"✅ Metadata extraction complete. Saved to: {output_path}")
