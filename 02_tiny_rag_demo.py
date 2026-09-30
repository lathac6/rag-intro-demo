import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

docs = [
    "Django uses an ORM to interact with the database.",
    "FastAPI is built on top of Starlette and Pydantic.",
    "PostgreSQL supports full-text search and JSON columns."
]

doc_embeddings = [
    genai.embed_content(model="models/gemini-embedding-001", content=d)["embedding"]
    for d in docs
]

def retrieve(query):
    q_emb = genai.embed_content(model="models/gemini-embedding-001", content=query)["embedding"]
    scores = [cosine_similarity(q_emb, d) for d in doc_embeddings]
    return docs[np.argmax(scores)]

question = "How does FastAPI relate to Pydantic?"
context = retrieve(question)

model = genai.GenerativeModel(
    "gemini-3.6-flash",
    system_instruction=f"Answer using only this context: {context}"
)

print(f"Retrieved context: {context}\n")
print(model.generate_content(question).text)