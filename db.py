


import os
import uuid
import fitz  # PyMuPDF
import re
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# --------------------------
# 🔧 Qdrant + Embedding Config
# --------------------------
QDRANT_URL = "https://c8df992d-b432-4052-b952-145841797199.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Z_FhOR0cy8m4lNqLU4x6d_IGaSx-Avhxn-piRXKgdFs"
COLLECTION_NAME = "insurance-docs"
EMBED_MODEL_NAME = "paraphrase-mpnet-base-v2"  # 768-dim output

# --------------------------
# ✅ Initialize
# --------------------------
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
model = SentenceTransformer(EMBED_MODEL_NAME)

# --------------------------
# 1. Create Qdrant collection (if not exists)
# --------------------------
def init_collection():
    if COLLECTION_NAME not in [c.name for c in qdrant.get_collections().collections]:
        qdrant.create_collection(
            COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"ℹ️ Collection already exists: {COLLECTION_NAME}")

# --------------------------
# 2. Extract text from PDF
# --------------------------
def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"❌ File not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# --------------------------
# 3. Smart chunking into 120-word blocks with overlap
# --------------------------
def split_into_chunks(text, max_words=120, overlap=30):
    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i:i + max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap  # slide forward with overlap

    return chunks

# --------------------------
# 4. Upload Chunks to Qdrant
# --------------------------
def upload_chunks(pdf_path):
    pdf_name = os.path.basename(pdf_path)
    print(f"\n📄 Processing {pdf_name}...")

    try:
        raw_text = extract_text_from_pdf(pdf_path)
    except Exception as e:
        print(str(e))
        return

    chunks = split_into_chunks(raw_text, max_words=120, overlap=30)
    print(f"🧩 Found {len(chunks)} chunks.")

    embeddings = model.encode(chunks, batch_size=8, show_progress_bar=True)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text": chunk,
                "source": pdf_name,
                "chunk_index": i
            }
        )
        for i, (chunk, vec) in enumerate(zip(chunks, embeddings))
    ]

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"✅ Uploaded {len(points)} chunks from {pdf_name}.")

# --------------------------
# 5. Run Script
# --------------------------
if __name__ == "__main__":
    init_collection()

    pdf_paths = [
        "data/EDLHLGA23009V012223.pdf",
        "data/ICIHLIP22012V012223.pdf",
        "data/HDFHLIP23024V072223.pdf",
        "data/BAJHLIP23020V012223.pdf",
        "data/CHOTGDP23004V012223.pdf"
    ]

    for path in pdf_paths:
        upload_chunks(path)
