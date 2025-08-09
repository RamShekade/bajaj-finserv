# import os
# import uuid
# import fitz  # PyMuPDF
# import numpy as np
# from tqdm import tqdm
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct
# from sentence_transformers import SentenceTransformer

# # --------------------------
# # 🔧 Qdrant + Embedding Config
# # --------------------------
# QDRANT_URL = "https://c8df992d-b432-4052-b952-145841797199.us-east4-0.gcp.cloud.qdrant.io:6333"
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Z_FhOR0cy8m4lNqLU4x6d_IGaSx-Avhxn-piRXKgdFs"
# COLLECTION_NAME = "insurance"
# EMBED_MODEL_NAME = "paraphrase-MiniLM-L6-v2"  # 384-dim sentence-transformers embeddings

# # --------------------------
# # ✅ Initialize
# # --------------------------
# qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
# embedder = SentenceTransformer(EMBED_MODEL_NAME)
# EMBED_DIM = embedder.get_sentence_embedding_dimension()

# # --------------------------
# # 1. Create Qdrant collection (if not exists)
# # --------------------------
# def init_collection():
#     if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
#         qdrant.create_collection(
#             COLLECTION_NAME,
#             vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
#         )
#         print(f"✅ Created collection: {COLLECTION_NAME}")
#     else:
#         print(f"ℹ️ Collection already exists: {COLLECTION_NAME}")

# # --------------------------
# # 2. Extract text from PDF
# # --------------------------
# def extract_text_from_pdf(pdf_path):
#     if not os.path.exists(pdf_path):
#         raise FileNotFoundError(f"❌ File not found: {pdf_path}")
#     doc = fitz.open(pdf_path)
#     return "\n".join([page.get_text() for page in doc])

# # --------------------------
# # 3. Smart chunking into 120-word blocks with overlap
# # --------------------------
# def split_into_chunks(text, max_words=120, overlap=30):
#     words = text.split()
#     chunks = []
#     i = 0
#     while i < len(words):
#         chunk = words[i:i + max_words]
#         chunks.append(" ".join(chunk))
#         i += max_words - overlap  # slide forward with overlap
#     return chunks

# # --------------------------
# # 4. Get embedding for a chunk using sentence-transformers
# # --------------------------
# def get_chunk_embedding(text):
#     emb = embedder.encode([text], normalize_embeddings=True)
#     return emb[0].astype(np.float32)

# # --------------------------
# # 5. Upload Chunks to Qdrant
# # --------------------------
# def upload_chunks(pdf_path):
#     pdf_name = os.path.basename(pdf_path)
#     print(f"\n📄 Processing {pdf_name}...")

#     try:
#         raw_text = extract_text_from_pdf(pdf_path)
#     except Exception as e:
#         print(str(e))
#         return

#     chunks = split_into_chunks(raw_text, max_words=120, overlap=30)
#     print(f"🧩 Found {len(chunks)} chunks.")

#     embeddings = []
#     for chunk in tqdm(chunks, desc="Embedding Chunks"):
#         emb = get_chunk_embedding(chunk)
#         embeddings.append(emb.tolist())

#     points = [
#         PointStruct(
#             id=str(uuid.uuid4()),
#             vector=vec,
#             payload={
#                 "text": chunk,
#                 "source": pdf_name,
#                 "chunk_index": i
#             }
#         )
#         for i, (chunk, vec) in enumerate(zip(chunks, embeddings))
#     ]

#     qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
#     print(f"✅ Uploaded {len(points)} chunks from {pdf_name}.")

# # --------------------------
# # 6. Run Script
# # --------------------------
# if __name__ == "__main__":
#     init_collection()

#     pdf_paths = [
#         "data/EDLHLGA23009V012223.pdf",
#         "data/ICIHLIP22012V012223.pdf",
#         "data/HDFHLIP23024V072223.pdf",
#         "data/BAJHLIP23020V012223.pdf",
#         "data/CHOTGDP23004V012223.pdf"
#     ]

#     for path in pdf_paths:
#         upload_chunks(path)


import json
import os
import uuid
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np

# Qdrant and embedding model config
QDRANT_URL = "https://c8df992d-b432-4052-b952-145841797199.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Z_FhOR0cy8m4lNqLU4x6d_IGaSx-Avhxn-piRXKgdFs"
COLLECTION_NAME = "insurance_queries"
EMBED_MODEL_NAME = "paraphrase-MiniLM-L6-v2"

# Initialize clients
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

def init_collection():
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"ℹ️ Collection already exists: {COLLECTION_NAME}")

def embed_and_upload(json_path):
    # Load data
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ File not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"📄 Loaded {len(data)} entries from {json_path}")

    # Prepare and upload
    points = []
    for idx, entry in enumerate(tqdm(data, desc="Embedding & Uploading")):
        query = entry["query"]
        answer = entry.get("answer", {})
        clauses = entry.get("clauses", [])
        vector = embedder.encode([query], normalize_embeddings=True)[0].astype(np.float32).tolist()
        payload = {
            "query": query,
            "answer": answer,
            "clauses": clauses,
            "index": idx
        }
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        )
        # Optional: batch upload every 1000 points for very large datasets
        if len(points) >= 1000:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []
    # Upload remaining
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ All queries uploaded to Qdrant!")

if __name__ == "__main__":
    init_collection()
    embed_and_upload("combined_policy_dataset.json")