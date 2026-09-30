from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import os
from dotenv import load_dotenv
from google import genai

# =========================
# LOAD ENVIRONMENT
# =========================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")

client = genai.Client(api_key=api_key)

# =========================
# FASTAPI
# =========================

app = FastAPI(
    title="AppleSupport AI Customer Support Agent",
    description="AI Customer Support using FAISS RAG + Gemini",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# LOAD FAISS
# =========================

FAISS_FILE = "data/processed/apple_support.faiss"
RAG_FILE = "data/processed/apple_support_labeled.csv"

index = faiss.read_index(FAISS_FILE)
rag_df = pd.read_csv(RAG_FILE)

print("FAISS loaded:", index.ntotal)
print("RAG data loaded:", len(rag_df))

# =========================
# LOAD EMBEDDING MODEL
# =========================

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded")

# =========================
# REQUEST FORMAT
# =========================

class CustomerQuery(BaseModel):
    message: str


# =========================
# RETRIEVAL
# =========================

def retrieve(query, top_k=5):

    embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        row = rag_df.iloc[idx]

        results.append({
            "customer_message": str(row["customer_message"]),
            "support_response": str(row["support_response"]),
            "intent": str(row["intent"]) if "intent" in row else "other",
            "similarity": round(float(score), 4)
        })

    return results


# =========================
# ESCALATION
# =========================

def decide_escalation(query, retrieved_results):

    query_lower = query.lower()

    sensitive_keywords = [
        "hacked",
        "fraud",
        "stolen",
        "lawsuit",
        "legal",
        "chargeback",
        "police",
        "security",
        "personal information"
    ]

    for keyword in sensitive_keywords:

        if keyword in query_lower:

            return (
                "ESCALATED",
                f"Sensitive issue detected: {keyword}"
            )

    if not retrieved_results:

        return (
            "ESCALATED",
            "No historical conversation found"
        )

    best_score = retrieved_results[0]["similarity"]

    if best_score < 0.60:

        return (
            "ESCALATED",
            "No sufficiently similar historical conversation found"
        )

    return (
        "AUTO-HANDLED",
        "Relevant historical conversations found"
    )


# =========================
# GEMINI RESPONSE
# =========================

def generate_response(query, retrieved_results):

    context = ""

    for i, result in enumerate(retrieved_results, start=1):

        context += f"""
Historical Conversation {i}

Customer:
{result["customer_message"]}

AppleSupport Response:
{result["support_response"]}

Similarity:
{result["similarity"]}
"""

    prompt = f"""
You are an Apple customer support assistant.

Answer the customer's question using the historical
AppleSupport conversations provided below.

Do not invent information.

If the historical conversations do not provide enough
information, give a safe and helpful response.

Customer Query:
{query}

Historical Conversations:
{context}

Give only the final customer-support response.
"""

    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini API generation note: {e}")
        if retrieved_results and "support_response" in retrieved_results[0]:
            return f"{retrieved_results[0]['support_response']}"
        return "Thank you for reaching out. We are reviewing your issue. For immediate assistance, please reach out to Apple Support or send us a direct message."


# =========================
# HEALTH CHECK
# =========================

@app.get("/")
def home():

    return {
        "status": "running",
        "message": "AppleSupport AI Customer Support Agent"
    }


# =========================
# CUSTOMER SUPPORT API
# =========================

@app.post("/support")
def support(query: CustomerQuery):

    customer_message = query.message.strip()

    if not customer_message:

        return {
            "error": "Customer message cannot be empty"
        }

    # 1. Retrieve
    retrieved = retrieve(
        customer_message,
        top_k=5
    )

    # 2. Escalation decision
    decision, reason = decide_escalation(
        customer_message,
        retrieved
    )

    # 3. Generate response
    ai_response = generate_response(
        customer_message,
        retrieved
    )

    # 4. Return complete result
    return {

        "customer_message": customer_message,

        "decision": decision,

        "reason": reason,

        "intent": retrieved[0]["intent"] if retrieved and "intent" in retrieved[0] else "other",
        "best_similarity": retrieved[0]["similarity"]
        if retrieved else 0,

        "generated_response": ai_response,

        "retrieved_conversations": retrieved
    }