from flask import Flask, request, jsonify
import faiss
import json
from sentence_transformers import SentenceTransformer
import openai
from collections import defaultdict

# Load FAISS index and text chunks
index = faiss.read_index('facstructure.index')
with open('text_chunks.json', 'r') as f:
    text_chunks = json.load(f)

# Initialize the sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

import os
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Flask
app = Flask(__name__)

# Track user messages using IP
user_message_count = defaultdict(int)
MESSAGE_LIMIT = 50  # 50 messages per month

@app.route('/chat', methods=['POST'])
def chat():
    user_ip = request.remote_addr  # Get user's IP address
    if user_message_count[user_ip] >= MESSAGE_LIMIT:
        return jsonify({"error": "Message limit reached for this month."}), 403

    # Get the query from the POST request
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query is required."}), 400

    # Step 1: Encode the query and find the closest text chunk
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, 1)
    closest_index = indices[0][0]

    # Step 2: Get the corresponding text chunk
    closest_text = text_chunks[closest_index]

    # Step 3: Generate the response using OpenAI
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Based on this text: {closest_text}, answer this question: {query}"}
        ]
    )
    answer = response['choices'][0]['message']['content']

    # Increment user's message count
    user_message_count[user_ip] += 1

    return jsonify({"answer": answer})

if __name__ == '__main__':
    app.run(debug=True)
