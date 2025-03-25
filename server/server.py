import os
import openai
import numpy as np
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import logging

logging.basicConfig(level=logging.INFO)

# Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Ensure OPENAI_API_KEY and PINECONE_API_KEY are set.")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "cooper-bot-index"

if INDEX_NAME not in pc.list_indexes().names():
    raise ValueError(f"Index '{INDEX_NAME}' does not exist.")
index = pc.Index(INDEX_NAME)

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Embedding function
def get_embedding(text):
    try:
        response = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[text],
            api_key=OPENAI_API_KEY
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Embedding Error: {e}")
        return None

# Pinecone query
def retrieve_relevant_qa(query_text, top_k=3):
    logging.info(f"Retrieving info for: {query_text}")
    embedding = get_embedding(query_text)
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
        logging.error(f"Pinecone Query Error: {e}")
        return [], []

# OpenAI chat completion
def query_openai(user_question, retrieved_context):
    context_str = "\n".join(retrieved_context) or "No relevant info found."
    prompt = f"""
You are Cooper Bot, trained to sound like Cooper Fruth.

Use the info below to help answer the user's question. Rephrase when needed, quote directly when fitting:

{context_str}

User Question: {user_question}
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
        logging.error(f"OpenAI Chat Error: {e}")
        return "I ran into an issue generating a response."

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Missing question."}), 400

    question = data["question"]
    context, scores = retrieve_relevant_qa(question)
    answer = query_openai(question, context)

    return jsonify({"response": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
