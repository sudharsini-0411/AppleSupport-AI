import pandas as pd
import numpy as np
import faiss
import os
from sentence_transformers import SentenceTransformer

# -----------------------------
# File paths
# -----------------------------
input_file = "data/processed/apple_support_clean.csv"
index_file = "data/processed/apple_support.faiss"
data_file = "data/processed/apple_support_rag.csv"

# -----------------------------
# Load dataset
# -----------------------------
print("Loading dataset...")

df = pd.read_csv(input_file)

df = df.dropna(subset=["customer_message", "support_response"])

print("Conversations:", len(df))

# -----------------------------
# Load embedding model
# -----------------------------
print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded!")

# -----------------------------
# Create embeddings
# -----------------------------
print("\nCreating embeddings...")

embeddings = model.encode(
    df["customer_message"].tolist(),
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

print("Embedding shape:", embeddings.shape)

# -----------------------------
# Create FAISS index
# -----------------------------
print("\nCreating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

print("FAISS vectors:", index.ntotal)

# -----------------------------
# Save files
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

faiss.write_index(index, index_file)

df.to_csv(data_file, index=False)

print("\n==============================")
print("FAISS RAG database created!")
print("==============================")
print("Index:", index_file)
print("Data :", data_file)