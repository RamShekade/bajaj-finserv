import os
import requests
from flask import Flask, request, jsonify
from qdrant_client import QdrantClient
import fitz  # PyMuPDF
import uuid
import numpy as np
import google.generativeai as genai
import re
import faiss
from sentence_transformers import SentenceTransformer

# ---- CONFIG ----
QDRANT_URL = "https://c8df992d-b432-4052-b952-145841797199.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Z_FhOR0cy8m4lNqLU4x6d_IGaSx-Avhxn-piRXKgdFs"
COLLECTION_NAME = "insurance"
GEMINI_API_KEY = "AIzaSyBcieQSbcnDkWnxcRyHKusdp5-TQWK-5Fs"
EXPECTED_API_KEY = "74915bc2932a330cb216159be4298485ee0af534a2d30eacee1be61d2158d6b2"

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

embedder = SentenceTransformer("paraphrase-MiniLM-L6-v2")
EMBED_DIM = embedder.get_sentence_embedding_dimension()

app = Flask(__name__)

def search_qdrant(query, top_k=8):
    query_emb = get_text_embedding(query)
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_emb.tolist(),
        limit=top_k
    )
    results = []
    for hit in hits:
        payload = hit.payload
        results.append({
            "text": payload.get("text", ""),
            "clause": payload.get("clause", None)
        })
    return results

def extract_text_from_pdf(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text() + "\n"
    return text

def clause_level_chunking(text, min_words=20):
    """
    Improved chunking: splits on clause/section numbers and bullet points.
    Each chunk is a clause or section (with overlap for context).
    """
    # Use regex to split at clause/section headers (customize as needed)
    split_regex = r"(?=([A-Z]?\d{1,2}(\.[a-zA-Z0-9]+)*\.?\s)|(\n\s*-\s)|(\n\s*\u2022\s))"
    candidates = re.split(split_regex, text)
    # Recombine matches into chunks
    chunks = []
    chunk = ""
    for part in candidates:
        if part is None:
            continue
        if part.strip() == "":
            continue
        if re.match(r"^([A-Z]?\d{1,2}(\.[a-zA-Z0-9]+)*\.?)$", part.strip()):
            if chunk.strip():
                chunks.append(chunk.strip())
            chunk = part
        else:
            chunk += " " + part
    if chunk.strip():
        chunks.append(chunk.strip())
    # Filter out very small chunks
    chunks = [c for c in chunks if len(c.split()) >= min_words]
    # Add overlap (previous chunk) for context
    overlapped_chunks = []
    prev = ""
    for c in chunks:
        if prev:
            overlapped_chunks.append(prev + " " + c)
        else:
            overlapped_chunks.append(c)
        prev = c
    return overlapped_chunks

def get_clause_number(chunk):
    # Try to extract clause/section number from beginning of chunk
    match = re.match(r"([A-Z]?\d{1,2}(\.[a-zA-Z0-9]+)*)", chunk.strip())
    return match.group(0) if match else None

def get_text_embedding(text):
    # Use sentence-transformers for strong semantic retrieval
    emb = embedder.encode([text], normalize_embeddings=True)
    return emb[0].astype(np.float32)

class InMemoryFAISSIndex:
    def __init__(self, dim):
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []
        self.clauses = []
        self.chunk_indices = []

    def build(self, chunks):
        self.index.reset()
        self.chunks = []
        self.clauses = []
        self.chunk_indices = []
        if not chunks:
            return
        vectors = []
        for i, chunk in enumerate(chunks):
            vec = get_text_embedding(chunk)
            vectors.append(vec)
            self.chunks.append(chunk)
            self.clauses.append(get_clause_number(chunk))
            self.chunk_indices.append(i)
        arr = np.stack(vectors)
        self.index.add(arr)

    def search(self, query, top_k=8):
        q_emb = get_text_embedding(query)
        q_emb = np.expand_dims(q_emb, axis=0)
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx >= 0 and idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "clause": self.clauses[idx]
                })
        return results

faiss_index = InMemoryFAISSIndex(EMBED_DIM)

def call_gemini_flash(query, relevant_chunks, source="document"):
    if source == "document":
        intro = "Here are the most relevant clauses from the uploaded document"
    else:
        intro = "Here are the most relevant pieces of knowledge from the knowledge base"

    prompt = f"""You are an expert insurance policy assistant.
The user asked: "{query}"

{intro} (with clause/section numbers when available):

"""
    for i, chunk in enumerate(relevant_chunks):
        clause = chunk.get("clause")
        clause_info = f"(Clause {clause}) " if clause else ""
        prompt += f"{i+1}. {clause_info}{chunk['text'].strip()}\n\n"

    prompt += """\
Instructions:
- Extract the answer ONLY from the provided pieces above. Do not use outside knowledge unless the answer is truly not present.
- If the answer is not present, reply: "The answer is not present in the knowledge base."
- When possible, cite the clause/section number in your answer.
- Be concise, clear, and professional.
- Return your answer as plain text only (no Markdown, no bullet points, no extra formatting).
"""
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

def ensure_doc_indexed(text):
    # Improved clause-level chunking and in-memory FAISS indexing
    chunks = clause_level_chunking(text)
    faiss_index.build(chunks)

def answer_questions(questions):
    answers = []
    for q in questions:
        doc_chunks = faiss_index.search(q, top_k=8)
        answer = call_gemini_flash(q, doc_chunks, source="document")
        # Accept both "the document does not mention" and "the answer is not present in the knowledge base"
        if answer.strip().lower().startswith("the document does not mention") or \
           answer.strip().lower().startswith("the answer is not present"):
            # fallback to knowledge base
            kb_chunks = search_qdrant(q, top_k=8)
            kb_answer = call_gemini_flash(q, kb_chunks, source="knowledge_base")
            answers.append(kb_answer)
        else:
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
        text = get_pdf_text(pdf_url)
        ensure_doc_indexed(text)
    except Exception as e:
        return jsonify({"error": f"Could not process document: {str(e)}"}), 400

    answers = answer_questions(questions)
    return jsonify({"answers": answers})

@app.route("/", methods=["GET"])
def home():
    return {"status": "ok"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)