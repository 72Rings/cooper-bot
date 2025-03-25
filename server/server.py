# server/server.py

import os
import openai
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import logging

logging.basicConfig(level=logging.INFO)

# Load API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
openai.api_key = OPENAI_API_KEY

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("API keys not set in environment variables.")

# Init Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "cooper-bot-index"
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist.")
index = pc.Index(INDEX_NAME)

# Flask app
app = Flask(__name__)
CORS(app)

# Embedding
def get_embedding(text):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding error: {e}")
        return None

# Pinecone Query
def retrieve_context(question):
    embedding = get_embedding(question)
    if embedding is None:
        return []

    try:
        results = index.query(vector=embedding, top_k=3, include_metadata=True)
        return [
            f"Q: {m['metadata']['question']}\nA: {m['metadata']['answer']}"
            for m in results["matches"]
        ]
    except Exception as e:
        logging.error(f"Pinecone query error: {e}")
        return []

# Chat generation
def ask_openai(question, context):
    prompt = f"""
You are Cooper Bot. Use the info below to answer naturally.

{chr(10).join(context)}

User: {question}
"""
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI response error: {e}")
        return "I hit a problem responding. Try again later."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question"}), 400

    q = data["question"]
    context = retrieve_context(q)
    answer = ask_openai(q, context)
    return jsonify({"response": answer})

# Use Railway-required PORT
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # 👈 fallback changed to 5000
    app.run(host="0.0.0.0", port=port)

