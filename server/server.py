import os
import logging
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from pinecone import Pinecone
import openai

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    logging.error("Missing OPENAI_API_KEY or PINECONE_API_KEY environment variable.")
    raise ValueError("API keys not set in environment variables.")

# Initialize OpenAI client
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "cooper-bot-index"

if INDEX_NAME not in pc.list_indexes().names():
    logging.error(f"Pinecone index '{INDEX_NAME}' does not exist.")
    raise ValueError(f"Pinecone index '{INDEX_NAME}' does not exist.")

index = pc.Index(INDEX_NAME)

# Initialize Flask app
app = Flask(__name__)
CORS(app)


# OpenAI embedding function
def get_embedding(text):
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None


# Query Pinecone
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
    system_message = f"""
You are Cooper Bot, trained to replicate Cooper Fruth's tone, humor, and conversational style.

Below are relevant pieces of information retrieved from past conversations:

{context_str}

Now, answer the user's question in Cooper's natural tone and style.
User Question: {user_question}
""".strip()

    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "system", "content": system_message}],
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"OpenAI Chat Error: {e}")
        return "I encountered an issue generating a response."


# Flask route
@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' in request."}), 400

        user_question = data["question"].strip()
        logging.info(f"Received question: {user_question}")

        retrieved_context, match_scores = retrieve_relevant_qa(user_question)
        bot_response = query_openai(user_question, retrieved_context)

        return jsonify({"response": bot_response})
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

