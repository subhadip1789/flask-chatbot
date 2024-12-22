from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
import json

# Step 1: Read and extract text from the PDF
pdf_file_path = 'FACstructure.pdf'  # Your PDF file name
pdf_reader = PdfReader(pdf_file_path)
pdf_text = ""

# Extract text from each page
for page in pdf_reader.pages:
    pdf_text += page.extract_text()

# Step 2: Break text into small chunks (split by line breaks)
text_chunks = pdf_text.split('\n')

# Save the text chunks to a file for future use
with open('text_chunks.json', 'w') as f:
    json.dump(text_chunks, f)

# Step 3: Use a small AI model to encode the text chunks into embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(text_chunks)

# Step 4: Save these embeddings in a searchable database (FAISS)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the index after adding embeddings
faiss.write_index(index, 'facstructure.index')
print("FAISS index saved successfully.")

print("Text chunks and index are ready for the chatbot!")
