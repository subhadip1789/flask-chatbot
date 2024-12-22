import openai
from sentence_transformers import SentenceTransformer
import faiss
import json

# Set your OpenAI API key
openai.api_key = 'sk-XXXXXXXXXXXXXXXXXXXXX'  # Replace with your actual API key

# Load the FAISS index
index = faiss.read_index('facstructure.index')  # Assuming you've saved the index as facstructure.index

# Load SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the text chunks (these should match the embeddings)
with open('facstructure_chunks.json', 'r') as f:
    text_chunks = json.load(f)

# Function to ask questions based on the text
def ask_question(query):
    # Generate the embedding for the query
    query_embedding = model.encode([query])
    
    # Find the closest text chunk using FAISS
    _, closest_index = index.search(query_embedding, 1)
    closest_text = text_chunks[closest_index[0][0]]  # Fetch the closest text
    
    # Use OpenAI's Chat API to answer the question based on the closest text
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Use the correct chat model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Based on this text: {closest_text}, answer this question: {query}"}
        ]
    )
    
    # Return the response
    return response['choices'][0]['message']['content']

# Example question
question = "What is the first lesson in the Introduction section?"
print(ask_question(question))
