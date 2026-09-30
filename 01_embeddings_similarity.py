import os
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    result = genai.embed_content(model="models/gemini-embedding-001", content=text)
    return result["embedding"]

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

pairs = [
    ("I love backend development", "I enjoy building server-side applications"),
    ("The cat sat on the mat", "A feline rested on the rug"),
    ("I love backend development", "The weather in Paris is rainy today"),
]

for text1, text2 in pairs:
    vec1 = get_embedding(text1)
    vec2 = get_embedding(text2)
    score = cosine_similarity(vec1, vec2)
    print(f"'{text1}' vs '{text2}' -> Similarity: {score:.4f}")