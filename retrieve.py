import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load FAISS index
index = faiss.read_index(
    "data/processed/apple_support.faiss"
)

# Load conversation data
df = pd.read_csv(
    "data/processed/apple_support_rag.csv"
)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Customer query
query = input("\nEnter customer message: ")

# Convert query to embedding
query_embedding = model.encode(
    [query],
    normalize_embeddings=True
)

# Search top 5 similar conversations
scores, indices = index.search(
    query_embedding,
    5
)

print("\n================================")
print("TOP 5 SIMILAR CONVERSATIONS")
print("================================")

for rank, (score, idx) in enumerate(
    zip(scores[0], indices[0]), start=1
):

    print(f"\n--- Result {rank} ---")
    print("Similarity:", round(float(score), 4))
    print("Customer:", df.iloc[idx]["customer_message"])
    print("Support :", df.iloc[idx]["support_response"])