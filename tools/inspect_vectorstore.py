import faiss
import pickle
from pathlib import Path

# Paths to your FAISS and Pickle files
vectorstore_dir = Path("../data/vectorstore")
faiss_file = vectorstore_dir / "index.faiss"
pkl_file = vectorstore_dir / "index.pkl"

# Load FAISS index
index = faiss.read_index(str(faiss_file))
print("✅ FAISS index loaded.")

# Inspect first 2 embeddings
for i in range(2):
    vec = index.reconstruct(i)
    print(f"🔢 Vector {i} (first 10 values): {vec[:10]}")

# Load the pickle (index_to_docstore_id, docstore)
with open(pkl_file, "rb") as f:
    index_to_docstore_id, docstore = pickle.load(f)

# Handle docstore that contains either strings or Document objects
print("\n📂 Inspecting docstore content...")
for i, doc in list(docstore.items())[:2]:
    print(f"\n📄 Document ID: {i}")
    if hasattr(doc, "page_content"):
        print(f"Text (preview): {doc.page_content[:200]}...")
        print(f"Metadata: {doc.metadata}")
    else:
        print(f"Raw Value (type: {type(doc)}): {doc[:200]}...")
