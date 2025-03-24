'''

import os
import openai
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone

# Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Ensure OPENAI_API_KEY and PINECONE_API_KEY are set.")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "cooper-bot-index"

# Ensure Pinecone index exists
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{INDEX_NAME}' does not exist. Run 'generate_embeddings.py' first.")

# Get reference to index
index = pc.Index(INDEX_NAME)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# OpenAI embedding function
def get_embedding(text):
    """Generate embedding using OpenAI API."""
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# Query Pinecone for similar results
def retrieve_relevant_qa(query_text, top_k=3):
    """Retrieve top relevant Q&A pairs from Pinecone."""
    embedding = get_embedding(query_text)
    if embedding is None:
        return "No relevant info found."

    results = index.query(vector=embedding, top_k=top_k, include_metadata=True)

    retrieved_context = []
    for match in results["matches"]:
        if "question" in match["metadata"] and "answer" in match["metadata"]:
            retrieved_context.append(f"Q: {match['metadata']['question']}\nA: {match['metadata']['answer']}")

    return "\n".join(retrieved_context) if retrieved_context else "No relevant info found."

# Query OpenAI function
def query_openai(prompt):
    """Query OpenAI's GPT model with retrieved context."""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return "I encountered an issue generating a response."

# Flask API Route
@app.route("/chat", methods=["POST"])
def chat():
    """Handles user queries, retrieves data from Pinecone, and sends response via OpenAI."""
    data = request.get_json()
    
    if not data or "question" not in data:
        return jsonify({"error": "Input text cannot be empty."}), 400

    user_question = data["question"].strip()
    retrieved_context = retrieve_relevant_qa(user_question)

    system_message = f"""
    You are Cooper Bot, trained to replicate Cooper Fruth's tone, humor, and conversational style.

    Below are relevant pieces of information retrieved from past conversations:

    {retrieved_context}

    Now, answer the user's question in Cooper's natural tone and style.
    User Question: {user_question}
    """

    bot_response = query_openai(system_message)
    return jsonify({"response": bot_response})

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
'''

import os
import openai
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone

# Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Ensure OPENAI_API_KEY and PINECONE_API_KEY are set.")

# Initialize Pinecone client
try:
    print("✅ Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    INDEX_NAME = "cooper-bot-index"
    
    if INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Index '{INDEX_NAME}' does not exist. Run 'generate_embeddings.py' first.")

    index = pc.Index(INDEX_NAME)
    print("✅ Pinecone connection successful.")
except Exception as e:
    print(f"❌ Pinecone Initialization Error: {e}")
    exit(1)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/chat": {"origins": "*"}}, supports_credentials=True)
print("✅ Flask server initialized.")

# OpenAI embedding function
def get_embedding(text):
    """Generate embedding using OpenAI API."""
    try:
        print(f"🔎 Generating embedding for: {text[:50]}...")
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return None

# Query Pinecone for similar results
def retrieve_relevant_qa(query_text, top_k=3):
    """Retrieve top relevant Q&A pairs from Pinecone."""
    print(f"🔍 Searching Pinecone for: {query_text[:50]}")
    embedding = get_embedding(query_text)
    if embedding is None:
        return None, None  # Return empty results

    try:
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        retrieved_context = []
        match_scores = []
        
        for match in results["matches"]:
            if "question" in match["metadata"] and "answer" in match["metadata"]:
                retrieved_context.append(f"Q: {match['metadata']['question']}\nA: {match['metadata']['answer']}")
                match_scores.append(match["score"])  # Save match score
        
        print(f"📌 Found {len(retrieved_context)} relevant results.")
        return retrieved_context, match_scores if retrieved_context else (None, None)
    except Exception as e:
        print(f"❌ Pinecone Query Error: {e}")
        return None, None

# Query OpenAI function
def query_openai(user_question, retrieved_context, match_scores):
    """Query OpenAI's GPT model with retrieved context and enhanced logic."""
    context_str = "\n".join(retrieved_context) if retrieved_context else "No relevant info found."

    system_message = f"""
    You are Cooper Bot, trained to replicate Cooper Fruth's tone, humor, and conversational style.

    Below are relevant pieces of information retrieved from past conversations:

    {context_str}

    === Response Guidelines ===
    - If a retrieved answer is **highly relevant**, quote **as much as possible** to stay authentic.
    - If multiple pieces of relevant information exist, blend them into a **coherent answer**.
    - If the retrieved information isn't a perfect match, use it as guidance but **rephrase naturally**.
    - If no relevant context is available, generate a response using your training.

    User Question: {user_question}
    """

    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": system_message}],
            temperature=0.7
        )
        print("✅ OpenAI Response Generated.")
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return "I encountered an issue generating a response."

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        print("📩 Received request data:", data)

        if not data or "question" not in data:
            return jsonify({"error": "Invalid request. Expected a 'question' key."}), 400

        user_question = data["question"].strip()
        session_id = data.get("session_id", "default")
        print(f"👤 Session ID: {session_id}")

        retrieved_context, match_scores = retrieve_relevant_qa(user_question)
        bot_response = query_openai(user_question, retrieved_context, match_scores)

        response = jsonify({"response": bot_response})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    except Exception as e:
        print(f"❌ Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    print("🚀 Starting Flask Server on port 5000...")
    app.run(host="127.0.0.1", port=5000, debug=True)
