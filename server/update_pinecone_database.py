import os
import openai
from pinecone import Pinecone, ServerlessSpec
import json
import numpy as np

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define index name
index_name = "cooper-bot-index"

# Ensure the Pinecone index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index '{index_name}' already exists.")

# OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate embeddings
def get_embedding(text):
    """Generate an embedding using OpenAI's latest API."""
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=[text]
        )
        return response["data"][0]["embedding"]
    except Exception as e:
        print(f"\u274c Error generating embedding: {e}")
        return None

# Load Q&A pairs from the JSONL file and process them
def process_and_upsert(jsonl_file):
    """Extract Q&A pairs and insert them into Pinecone."""
    index = pc.Index(index_name)
    
    with open(jsonl_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    new_entries = 0
    for item in data:
        question = None
        answer = None

        # Extract question and answer
        for message in item["messages"]:
            if message["role"] == "user":
                question = message["content"]
            elif message["role"] == "assistant":
                answer = message["content"]

        # Validate the extracted values
        if not question or not answer:
            continue  # Skip entries missing a question or answer

        # Generate embedding for the question
        embedding = get_embedding(question)
        if embedding is None:
            continue  # Skip if embedding generation failed

        # Prepare the Pinecone upsert payload
        doc_id = str(hash(question))  # Unique ID for the question
        vector = embedding  # Embedding as a list
        metadata = {"question": question, "answer": answer}

        try:
            index.upsert(vectors=[{"id": doc_id, "values": vector, "metadata": metadata}])
            new_entries += 1
            print(f"Upserted Q&A pair with ID {doc_id}")
        except Exception as e:
            print(f"\u274c Error upserting Q&A pair with ID {doc_id}: {e}")

    print(f"\u2705 Successfully added {new_entries} Q&A pairs to the Pinecone index.")

# Run the script to process the file and update the Pinecone index
if __name__ == "__main__":
    process_and_upsert("test_run_embeddings.jsonl")
