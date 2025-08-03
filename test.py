import os
import fitz  # PyMuPDF
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

# Config
QDRANT_URL = "https://c8df992d-b432-4052-b952-145841797199.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.Z_FhOR0cy8m4lNqLU4x6d_IGaSx-Avhxn-piRXKgdFs"
COLLECTION_NAME = "insurance-docs"

# Initialize
model = SentenceTransformer("paraphrase-mpnet-base-v2")  # Fast & light model
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def search_qdrant(query, top_k=5):
    query_vector = model.encode(query).tolist()

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k
    )

    for res in results:
        print(f"\nScore: {res.score:.4f}")
        print(f"Source: {res.payload['source']}")
        print(f"Text: {res.payload['text'][:300]}...\n")

# Example
search_qdrant("Does this policy provide preventive health checkup benefits?")


