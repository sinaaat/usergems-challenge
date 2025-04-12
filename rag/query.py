from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# Load vector store
vectorstore_path = Path("../data/vectorstore")
embedding_model = OpenAIEmbeddings()

vectorstore = FAISS.load_local(
    str(vectorstore_path),
    embedding_model,
    allow_dangerous_deserialization=True
)
print("✅ Vector store loaded.")

# Set up GPT-4o model
llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

# Create QA chain with retriever
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
    return_source_documents=True
)

# User query
query = input("\n📝 Enter your question: ")

# Get answer
result = qa_chain(query)

# Show answer
print("\n🧠 GPT-4o Answer:")
print(result["result"])

# Optional: show source chunks used
print("\n📚 Retrieved Chunks Used:")
for i, doc in enumerate(result["source_documents"]):
    print(f"Source {i+1}: {doc.metadata.get('company_name')} (chunk {doc.metadata.get('chunk_id')})")
    print(f"{doc.page_content[:200]}...")
    print("-" * 60)
