import os
import requests
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
import fitz  # PyMuPDF
import uuid
import gensim.downloader as api
import numpy as np
import google.generativeai as genai
import re

# ---- CONFIG ----
QDRANT_URL = "https://c8df992d-b432-4052-b952-145841797199.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Z_FhOR0cy8m4lNqLU4x6d_IGaSx-Avhxn-piRXKgdFs"
COLLECTION_NAME = "insurance-docs"
GEMINI_API_KEY = "AIzaSyAXMJzR77i8gq0XAqIn-15rHHuyVfgSqSs"
EXPECTED_API_KEY = "74915bc2932a330cb216159be4298485ee0af534a2d30eacee1be61d2158d6b2"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

wv = api.load("glove-wiki-gigaword-50")  # 50d vectors

app = Flask(__name__)

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    paragraphs = []
    for page in doc:
        text = page.get_text()
        lines = text.split('\n')
        buffer = []
        for line in lines:
            if line.strip() == "":
                if buffer:
                    paragraphs.append(" ".join(buffer).strip())
                    buffer = []
            else:
                buffer.append(line.strip())
        if buffer:
            paragraphs.append(" ".join(buffer).strip())
    return paragraphs

def chunk_paragraphs(paragraphs, max_words=200):
    chunks = []
    current_chunk = []
    current_len = 0
    for para in paragraphs:
        length = len(para.split())
        if current_len + length > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0
        current_chunk.append(para)
        current_len += length
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def get_clause_number(chunk):
    match = re.search(r'([0-9]{1,2}(\.[0-9]{1,2})*)', chunk)
    return match.group(0) if match else None

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
            payload={
                "text": chunk,
                "source": source_name,
                "chunk_index": i,
                "clause": get_clause_number(chunk)
            }
        )
        for i, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    client.upsert(collection_name=COLLECTION_NAME, points=points)

def search_qdrant_with_fallback(query, doc_source_name, top_k=5):
    query_vec = get_text_embedding(query).tolist()
    # Get more results to filter manually
    results_doc = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k * 5,  # get enough to filter
    )
    # Manually filter for source_name
    doc_chunks = [
        {'text': r.payload["text"], 'clause': r.payload.get("clause")}
        for r in results_doc if r.payload.get("source") == doc_source_name
    ]
    doc_chunks = doc_chunks[:top_k]
    # Fallback to full DB if not found in doc
    fallback_chunks = None
    if not doc_chunks or all(not c['text'].strip() for c in doc_chunks):
        results_fallback = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
        )
        fallback_chunks = [
            {'text': r.payload["text"], 'clause': r.payload.get("clause")}
            for r in results_fallback
        ]
    return doc_chunks, fallback_chunks

def call_gemini_flash(query, relevant_chunks, fallback_chunks=None):
    prompt = f"""You are an insurance policy assistant.
The user asked: {query}

Here are relevant clauses from the uploaded document:
"""
    for i, chunk in enumerate(relevant_chunks):
        clause = chunk.get("clause")
        clause_info = f" (Clause {clause})" if clause else ""
        prompt += f"\n{str(i+1)}.{clause_info} {chunk['text']}"

    if fallback_chunks:
        prompt += "\n\nHere is additional context from other insurance documents in the database:"
        for i, chunk in enumerate(fallback_chunks):
            clause = chunk.get("clause")
            clause_info = f" (Clause {clause})" if clause else ""
            prompt += f"\n{str(i+1)}.{clause_info} {chunk['text']}"

    prompt += """
If the uploaded document provides a clear answer, cite it. If not, use the context from other documents and clearly state that the answer is based on general insurance knowledge. Always answer professionally."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def get_pdf_paragraphs(pdf_url):
    response = requests.get(pdf_url)
    response.raise_for_status()
    pdf_bytes = response.content
    return extract_text_from_pdf(pdf_bytes)

def ensure_doc_indexed(paragraphs, source_name):
    try:
        client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"size": wv.vector_size, "distance": "Cosine"}
    )
    chunks = chunk_paragraphs(paragraphs)
    add_chunks_to_qdrant(chunks, source_name)

def answer_questions(source_name, questions):
    answers = []
    for q in questions:
        doc_chunks, fallback_chunks = search_qdrant_with_fallback(q, source_name, top_k=5)
        answer = call_gemini_flash(q, doc_chunks, fallback_chunks)
        answers.append(answer)
    return answers

@app.route("/hackrx/run", methods=["POST"])
def hackrx_run():
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return jsonify({"error": "Missing or invalid Authorization header"}), 401
    api_key = auth.split(" ", 1)[1]
    if api_key != EXPECTED_API_KEY:
        return jsonify({"error": "Invalid API key"}), 403

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json"}), 415
    data = request.get_json()
    pdf_url = data.get("documents")
    questions = data.get("questions")
    if not pdf_url or not questions:
        return jsonify({"error": "Missing 'documents' or 'questions'"}), 400
    if not isinstance(questions, list):
        return jsonify({"error": "'questions' must be a list"}), 400

    try:
        paragraphs = get_pdf_paragraphs(pdf_url)
        ensure_doc_indexed(paragraphs, pdf_url)
    except Exception as e:
        return jsonify({"error": f"Could not process document: {str(e)}"}), 400

    answers = answer_questions(pdf_url, questions)
    return jsonify({"answers": answers})

@app.route("/", methods=["GET"])
def home():
    return {"status": "ok"}