import os
from flask import Flask, request, jsonify
from typing import List
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import fitz  # PyMuPDF
import uuid
import google.generativeai as genai

# ---- CONFIG ----
QDRANT_URL = "https://c8df992d-b432-4052-b952-145841797199.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Z_FhOR0cy8m4lNqLU4x6d_IGaSx-Avhxn-piRXKgdFs"
COLLECTION_NAME = "insurance-docs"
EMBED_MODEL = "paraphrase-mpnet-base-v2"
GEMINI_API_KEY = "AIzaSyBcieQSbcnDkWnxcRyHKusdp5-TQWK-5Fs"  # <-- replace this with your Gemini API key

# ---- INITIALIZE ----
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")  # or use "gemini-pro" if you have access

# ---- UTILS ----
def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    return " ".join(page.get_text() for page in doc)

def chunk_text(text, max_words=200):
    sentences = text.split(". ")
    chunks, chunk = [], []
    for sentence in sentences:
        chunk.append(sentence)
        if len(" ".join(chunk).split()) > max_words:
            chunks.append(". ".join(chunk))
            chunk = []
    if chunk:
        chunks.append(". ".join(chunk))
    return chunks

def add_chunks_to_qdrant(chunks, source_name):
    vectors = embedder.encode(chunks).tolist()
    from qdrant_client.models import PointStruct
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={"text": chunk, "source": source_name, "chunk_index": i}
        )
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search_qdrant(query, top_k=5):
    query_vec = embedder.encode([query])[0].tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
    )
    return [r.payload["text"] for r in results]

def call_gemini_flash(query: str, relevant_chunks: List[str]) -> dict:
    # Build prompt
    prompt = f"""You are an insurance policy assistant.
The user asked: {query}
Here are relevant clauses from the document:
""" + "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(relevant_chunks))
    prompt += """
Based on these clauses, answer the user's query. 
Return a concise JSON object containing:
- decision (Approved/Denied/Unclear)
- justification (reasoning from clauses)
- clause (reference to clause/section if present)
If unsure, say so in the justification."""

    # Send prompt to Gemini Flash
    try:
        response = gemini_model.generate_content(prompt)
        # Try to extract JSON from the response
        import json
        import re
        # Find the first {...} JSON object in the response
        match = re.search(r"\{.*?\}", response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        else:
            # If no JSON, return full text
            return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

# ---- FLASK APP ----
app = Flask(__name__)

@app.route("/query", methods=["POST"])
def query_endpoint():
    query = request.form.get("query") or (request.get_json() or {}).get("query")


    # Optionally process uploaded files (if any)
    if "files" in request.files:
        files = request.files.getlist("files")
        for file in files:
            pdf_bytes = file.read()
            text = extract_text_from_pdf(pdf_bytes)
            chunks = chunk_text(text)
            add_chunks_to_qdrant(chunks, file.filename)

    # Search Qdrant for relevant info
    relevant_chunks = search_qdrant(query, top_k=5)
    # Get decision from Gemini Flash
    result = call_gemini_flash(query, relevant_chunks)
    # Return structured JSON
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)