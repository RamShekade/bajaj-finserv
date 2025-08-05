import os
import requests
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
import fitz  # PyMuPDF
import uuid
import gensim.downloader as api
import numpy as np
import google.generativeai as genai

# ---- CONFIG ----
QDRANT_URL = "https://c8df992d-b432-4052-b952-145841797199.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Z_FhOR0cy8m4lNqLU4x6d_IGaSx-Avhxn-piRXKgdFs"
COLLECTION_NAME = "insurance-docs"
GEMINI_API_KEY = "AIzaSyAXMJzR77i8gq0XAqIn-15rHHuyVfgSqSs"
EXPECTED_API_KEY = "74915bc2932a330cb216159be4298485ee0af534a2d30eacee1be61d2158d6b2"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Download small word2vec model at runtime (done once)
wv = api.load("glove-wiki-gigaword-50")  # 50d vectors, ~70MB

app = Flask(__name__)

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

def get_text_embedding(text):
    words = [w for w in text.split() if w in wv]
    if words:
        return np.mean([wv[w] for w in words], axis=0)
    else:
        return np.zeros(wv.vector_size)

def add_chunks_to_qdrant(chunks, source_name):
    vectors = [get_text_embedding(chunk).tolist() for chunk in chunks]
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
    query_vec = get_text_embedding(query).tolist()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
    )
    return [r.payload["text"] for r in results]

def call_gemini_flash(query, relevant_chunks):
    prompt = f"""You are an insurance policy assistant.
The user asked: {query}
Here are relevant clauses from the document:
""" + "\n".join(f"{i+1}. {chunk}" for i, chunk in enumerate(relevant_chunks))
    prompt += """
Based on these clauses, answer the user's query. 
Return only the answer, and cite the clause or section if relevant."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def get_pdf_text(pdf_url):
    response = requests.get(pdf_url)
    response.raise_for_status()
    pdf_bytes = response.content
    return extract_text_from_pdf(pdf_bytes)

def ensure_doc_indexed(doc_text, source_name):
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": wv.vector_size, "distance": "Cosine"}
    )
    chunks = chunk_text(doc_text)
    add_chunks_to_qdrant(chunks, source_name)

def answer_questions(source_name, questions):
    answers = []
    for q in questions:
        relevant_chunks = search_qdrant(q, top_k=5)
        answer = call_gemini_flash(q, relevant_chunks)
        answers.append(answer)
    return answers

@app.route("/hackrx/run", methods=["GET","POST"])
def hackrx_run():
    # --- AUTH ---
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid Authorization header"}), 401
    api_key = auth.split(" ", 1)[1]
    if api_key != EXPECTED_API_KEY:
        return jsonify({"error": "Invalid API key"}), 403

    # --- REQUEST DATA ---
    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json()
    pdf_url = data.get("documents")
    questions = data.get("questions")
    if not pdf_url or not questions:
        return jsonify({"error": "Missing 'documents' or 'questions'"}), 400
    if not isinstance(questions, list):
        return jsonify({"error": "'questions' must be a list"}), 400

    # --- PROCESS DOCUMENT ---
    try:
        document_text = get_pdf_text(pdf_url)
        ensure_doc_indexed(document_text, pdf_url)
    except Exception as e:
        return jsonify({"error": f"Could not process document: {str(e)}"}), 400

    # --- ANSWER QUESTIONS ---
    answers = answer_questions(pdf_url, questions)
    return jsonify({"answers": answers})

@app.route("/", methods=["GET"])
def home():
    return {"status": "ok"}
