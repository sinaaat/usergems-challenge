import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os

from dotenv import load_dotenv

load_dotenv()

# Load enriched document file
input_path = Path('../data/cleaned_docs/enriched_with_metadata.jsonl')
with open(input_path, 'r', encoding='utf-8') as f:
    docs = [json.loads(line) for line in f]

# Set chunking strategy
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "!", "?", " ", ""]
)

# Wrap chunks in LangChain Documents
all_chunks = []

for doc in docs:
    chunks = text_splitter.split_text(doc["content"])
    metadata_base = {
        "company_name": doc["company_name"],
        "base_url": doc["base_url"],
        "business_model": doc.get("metadata_extracted", {}).get("business_model"),
        "price_hint": doc.get("metadata_extracted", {}).get("price_hint"),
        "pricing_mentioned": doc.get("metadata_extracted", {}).get("pricing_mentioned"),
    }

    for i, chunk in enumerate(chunks):
        metadata = {**metadata_base, "chunk_id": i}
        all_chunks.append(Document(page_content=chunk, metadata=metadata))

print(f"✅ Prepared {len(all_chunks)} total chunks.")

# Generate embeddings and store in FAISS
print("🔄 Generating embeddings and building vector store...")
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(all_chunks, embedding_model)

# Save the FAISS index
vectorstore_path = Path('../data/vectorstore')
vectorstore.save_local(str(vectorstore_path))
print(f"✅ Vector store saved to: {vectorstore_path.resolve()}")
