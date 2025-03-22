import os
from pinecone import Pinecone

# Load API Key
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("Missing PINECONE_API_KEY environment variable.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cooper-bot-index"

# Ensure Pinecone index exists
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' does not exist.")

# Get index reference
index = pc.Index(index_name)

# Delete all records in the index
print(f"🗑 Deleting all records in '{index_name}'...")
index.delete(delete_all=True)

print(f"✅ Successfully deleted all records from '{index_name}'.")
