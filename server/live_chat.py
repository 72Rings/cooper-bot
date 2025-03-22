'''
import os
import openai
from pinecone import Pinecone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Request
import uvicorn
import logging

# ----------------------------
# 🔑 Load API Keys
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Ensure OPENAI_API_KEY and PINECONE_API_KEY are set.")

# ----------------------------
# 🏗️ Initialize Pinecone
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cooper-bot-index"

# Ensure Pinecone index exists
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' does not exist.")

index = pc.Index(index_name)  # Get Pinecone index reference

# ----------------------------
# 🚀 Initialize FastAPI
# ----------------------------
app = FastAPI()

# 🌍 Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 📨 Request Model
# ----------------------------
class QueryRequest(BaseModel):
    question: str

# ----------------------------
# 🔎 Retrieve Relevant Data from Pinecone
# ----------------------------
def retrieve_relevant_qa(query):
    """Retrieve top relevant Q&A pairs from Pinecone and determine quoting logic."""
    try:
        query_embedding = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        ).data[0].embedding

        search_results = index.query(
            vector=query_embedding,
            top_k=3,  # Return top 3 relevant matches
            include_metadata=True
        )

        retrieved_context = []
        best_match_score = 0  # Track the highest match score

        for match in search_results["matches"]:
            score = match.get("score", 0)  # Retrieve match score
            best_match_score = max(best_match_score, score)

            if "question" in match["metadata"] and "answer" in match["metadata"]:
                retrieved_context.append({
                    "question": match["metadata"]["question"],
                    "answer": match["metadata"]["answer"],
                    "score": score
                })

        return retrieved_context, best_match_score

    except Exception as e:
        print(f"❌ Pinecone Retrieval Error: {e}")
        return [], 0

# ----------------------------
# 🤖 Query OpenAI LLM
# ----------------------------
def query_openai(user_question, retrieved_context, best_match_score):
    """Use retrieved Pinecone data + OpenAI to generate a response."""
    formatted_context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in retrieved_context])

    # Determine if bot should **directly quote** or **paraphrase**
    if best_match_score > 0.90:  # Very strong match, quote it directly
        system_message = f"""
''' 
import os
import openai
from pinecone import Pinecone
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import Request
import uvicorn
import logging

# ----------------------------
# 🔑 Load API Keys
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing API keys. Ensure OPENAI_API_KEY and PINECONE_API_KEY are set.")

# ----------------------------
# 🏗️ Initialize Pinecone
# ----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "cooper-bot-index"

# Ensure Pinecone index exists
if index_name not in pc.list_indexes().names():
    raise ValueError(f"Index '{index_name}' does not exist.")

index = pc.Index(index_name)  # Get Pinecone index reference

# ----------------------------
# 🚀 Initialize FastAPI
# ----------------------------
app = FastAPI()

# 🌍 Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# 📨 Request Model
# ----------------------------
class QueryRequest(BaseModel):
    question: str

# ----------------------------
# 🔎 Retrieve Relevant Data from Pinecone
# ----------------------------
def retrieve_relevant_qa(query):
    """Retrieve top relevant Q&A pairs from Pinecone and determine quoting logic."""
    try:
        query_embedding = openai.embeddings.create(
            model="text-embedding-3-small",
            input=[query]
        ).data[0].embedding

        search_results = index.query(
            vector=query_embedding,
            top_k=3,  # Return top 3 relevant matches
            include_metadata=True
        )

        retrieved_context = []
        best_match_score = 0  # Track the highest match score

        for match in search_results["matches"]:
            score = match.get("score", 0)  # Retrieve match score
            best_match_score = max(best_match_score, score)

            if "question" in match["metadata"] and "answer" in match["metadata"]:
                retrieved_context.append({
                    "question": match["metadata"]["question"],
                    "answer": match["metadata"]["answer"],
                    "score": score
                })

        return retrieved_context, best_match_score

    except Exception as e:
        print(f"❌ Pinecone Retrieval Error: {e}")
        return [], 0

# ----------------------------
# 🤖 Query OpenAI LLM
# ----------------------------
def query_openai(user_question, retrieved_context, best_match_score):
    """Use retrieved Pinecone data + OpenAI to generate a response."""
    formatted_context = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in retrieved_context])

    # Determine if bot should **directly quote** or **paraphrase**
    if best_match_score > 0.90:  # Very strong match, quote it directly
        system_message = f"""
        You are Cooper Bot, a perfect AI replica of Cooper Fruth. Answer the following exactly as he would.

        A question **identical or nearly identical** to this one was asked before, so respond **using the previous answer exactly**, unless the phrasing requires slight tweaks.

        User Question: {user_question}
        Relevant Context (Direct Quote): "{retrieved_context[0]['answer']}"
        """

    elif best_match_score > 0.70:  # Medium match, paraphrase the response
        system_message = f"""
        You are Cooper Bot, a perfect AI replica of Cooper Fruth. Use the following knowledge to **answer in his voice**.

        Context retrieved from past responses:
        {formatted_context}

        Now, answer the user's question in Cooper's **natural tone and style**, but do not quote directly—paraphrase it naturally.
        User Question: {user_question}
        """

    else:  # No strong match, use OpenAI's general knowledge
        system_message = f"""
        You are Cooper Bot, an AI trained to answer as Cooper Fruth would. No **strong** context match was found for this query, so answer it using **Cooper’s natural tone and humor**.

        User Question: {user_question}
        """

    # Query OpenAI
    try:
        response = openai.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_question}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"❌ OpenAI Error: {e}")
        return "I encountered an issue generating a response."

# ----------------------------
# 🚀 Chat Endpoint
# ----------------------------
@app.post("/chat")
async def chat(request: Request):
    """Handles user queries, retrieves data from Pinecone, and sends response via OpenAI."""
    data = await request.json()
    logging.info(f"Received request data: {data}")
    
    try:
        user_question = data.get("question", "").strip()
        if not user_question:
            raise HTTPException(status_code=400, detail="Input text cannot be empty.")

        # Retrieve relevant Q&A pairs from Pinecone
        retrieved_context, best_match_score = retrieve_relevant_qa(user_question)

        # Get response from OpenAI
        bot_response = query_openai(user_question, retrieved_context, best_match_score)

        return {"response": bot_response}
    
    except Exception as e:
        print(f"❌ Chat API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

# ----------------------------
# 🏁 Run FastAPI Server
# ----------------------------
if __name__ == "__main__":
    # Kill any existing processes on port 8080
    os.system("powershell Get-NetTCPConnection -LocalPort 8080 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }")
    
    # Start FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8080)
