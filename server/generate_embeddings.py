import os
import openai
import json
import numpy as np
from pinecone import Pinecone

# Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Ensure OPENAI_API_KEY and PINECONE_API_KEY are set.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cooper-bot-index"

# Ensure Pinecone index exists
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' does not exist.")

# Get index reference
index = pc.Index(index_name)

# OpenAI embedding function
def get_embedding(text):
    """Generate embedding using OpenAI API."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",  # Using latest OpenAI embedding model
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# Load Q&A pairs from file
def process_and_upsert(jsonl_file):
    """Extract unique Q&A pairs and insert into Pinecone."""
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    inserted_ids = set()
    new_entries = 0
    batch = []

    for item in data:
        question = item.get("text", "").split("\n")[0].replace("Question: ", "").strip()
        answer = item.get("text", "").split("\n")[1].replace("Answer: ", "").strip()

        if not question or not answer or question in inserted_ids:
            continue  # Skip duplicates

        embedding = get_embedding(question)
        if embedding is None:
            continue

        doc_id = str(hash(question))  # Unique ID for the question
        vector = np.array(embedding).tolist()
        metadata = {"question": question, "answer": answer}
        
        batch.append((doc_id, vector, metadata))
        inserted_ids.add(question)
        new_entries += 1

        # Upsert in batches of 100 to optimize performance
        if len(batch) >= 100:
            index.upsert(vectors=batch)
            batch = []

    # Final batch upsert
    if batch:
        index.upsert(vectors=batch)

    print(f"✅ Successfully added {new_entries} unique Q&A pairs to Pinecone.")

# Run embedding update
if __name__ == "__main__":
    process_and_upsert("formatted_qa.jsonl")
