import os
import openai
import numpy as np
from pinecone import Pinecone

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Ensure OPENAI_API_KEY and PINECONE_API_KEY are set.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Pinecone Index Name
INDEX_NAME = "cooper-bot-index"

# Ensure Pinecone index exists
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{INDEX_NAME}' does not exist. Run 'generate_embeddings.py' first.")

# Get reference to index
index = pc.Index(INDEX_NAME)

# OpenAI embedding function
def get_embedding(text):
    """Generate embedding using OpenAI API."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",  # Use latest embedding model
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# Query Pinecone for similar results
def query_pinecone(query_text, top_k=5):
    """Search Pinecone for the most relevant stored entries."""
    embedding = get_embedding(query_text)
    if embedding is None:
        print("❌ Failed to generate query embedding.")
        return None

    # Search Pinecone index
    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

    if "matches" not in results:
        print("⚠️ No results found.")
        return None

    return results["matches"]

# Run a test query
if __name__ == "__main__":
    query_text = input("Enter a question: ")
    results = query_pinecone(query_text, top_k=3)

    if results:
        print("\n🔍 Retrieved Documents:")
        for i, match in enumerate(results, 1):
            metadata = match["metadata"]
            print(f"\nResult {i}:")
            print(f"Question: {metadata.get('question', 'N/A')}")
            print(f"Answer: {metadata.get('answer', 'N/A')}")
            print(f"Score: {match['score']:.4f}")
    else:
        print("❌ No relevant documents found.")
