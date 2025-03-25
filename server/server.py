import os
import logging
import json
import numpy as np
import aiohttp
import asyncio

from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone

# Configure logging
logging.basicConfig(level=logging.INFO)

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "cooper-bot-index"

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing environment variables for OpenAI or Pinecone")

# Pinecone Setup
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist.")
index = pc.Index(INDEX_NAME)

# Flask App
app = Flask(__name__)
CORS(app)

# Async OpenAI Embedding
async def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": [text]
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload) as resp:
                result = await resp.json()
                return result["data"][0]["embedding"]
    except Exception as e:
        logging.error(f"❌ Embedding Error: {e}")
        return None

# Pinecone Retrieval
async def retrieve_relevant_qa(query_text, top_k=3):
    logging.info(f"🔍 Retrieving info for: {query_text}")
    embedding = await get_embedding(query_text)
    if embedding is None:
        return [], []

    try:
        results = index.query(vector=embedding, top_k=top_k, include_metadata=True)
        context = []
        scores = []
        for match in results["matches"]:
            meta = match.get("metadata", {})
            if "question" in meta and "answer" in meta:
                context.append(f"Q: {meta['question']}\nA: {meta['answer']}")
                scores.append(match.get("score", 0))
        return context, scores
    except Exception as e:
        logging.error(f"❌ Pinecone Query Error: {e}")
        return [], []

# Async OpenAI Chat
async def query_openai(user_question, context_lines):
    prompt = f"""
You are Cooper Bot, trained to replicate Cooper Fruth's tone, humor, and conversational style.

Below are relevant pieces of information retrieved from past conversations:

{'\n'.join(context_lines) or "No relevant info found."}

Now, answer the user's question in Cooper's natural tone and style.
User Question: {user_question}
""".strip()

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "system", "content": prompt}],
        "temperature": 0.5
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload) as resp:
                result = await resp.json()
                return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logging.error(f"❌ OpenAI Chat Error: {e}")
        return "I encountered an issue generating a response."

# Async Flask route using asyncio.run
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request."}), 400

        user_question = data["question"].strip()
        logging.info(f"👤 Received question: {user_question}")

        async def process():
            context, _ = await retrieve_relevant_qa(user_question)
            return await query_openai(user_question, context)

        response_text = asyncio.run(process())
        return jsonify({"response": response_text})

    except Exception as e:
        logging.error(f"❌ Server Error: {e}")
        return jsonify({"error": "Internal server error"}), 500

# Run app
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
