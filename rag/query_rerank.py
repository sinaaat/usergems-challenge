from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import re

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Load vector store
vectorstore_path = Path("../data/vectorstore")
embedding_model = OpenAIEmbeddings()

vectorstore = FAISS.load_local(
    str(vectorstore_path),
    embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ Vector store loaded.")

# Set up LLM for final answer
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# User input
query = input("\n📝 Enter your question: ")

# Step 1: Retrieve more chunks
initial_docs = vectorstore.similarity_search(query, k=10)

# Step 2: Rerank with GPT
reranked_docs = []
for doc in initial_docs:
    prompt = f"""
You are evaluating how relevant the following document chunk is to answering the question.

Question: "{query}"

Chunk:
\"\"\"{doc.page_content}\"\"\"

Score the chunk from 0 to 10 based on how helpful it is for answering the question. Respond with a single JSON object:
{{ "relevance_score": number }}
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant scoring chunk relevance."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r'\{.*\}', raw, re.DOTALL)
        if match:
            score_obj = json.loads(match.group(0))
            score = score_obj.get("relevance_score", 0)
        else:
            score = 0
    except Exception as e:
        print(f"⚠️ Error scoring chunk: {e}")
        score = 0

    reranked_docs.append((score, doc))

# Step 3: Sort and select top 3
reranked_docs = sorted(reranked_docs, key=lambda x: x[0], reverse=True)
top_docs = [doc for score, doc in reranked_docs[:3]]

# Step 4: Combine and generate final answer
context = "\n\n".join([doc.page_content for doc in top_docs])
final_prompt = f"""
Use the context below to answer the question as clearly as possible.

Context:
\"\"\"{context}\"\"\"

Question: "{query}"
Answer:
"""

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": final_prompt}
    ],
    temperature=0.2
)

# Output final answer
answer = response.choices[0].message.content.strip()
print("\n🧠 Final Answer:")
print(answer)

# Show sources
print("\n📚 Top 3 chunks used:")
for i, doc in enumerate(top_docs):
    print(f"Chunk {i+1} from {doc.metadata.get('company_name')}")
    print(doc.page_content[:250])
    print("-" * 60)
