import os
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from google import genai

from escalation import decide_escalation


# ==========================================
# 1. Load Gemini API
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)


# ==========================================
# 2. Load FAISS and RAG data
# ==========================================

index = faiss.read_index(
    "data/processed/apple_support.faiss"
)

df = pd.read_csv(
    "data/processed/apple_support_rag.csv"
)

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("RAG system loaded successfully!")


# ==========================================
# 3. Retrieve similar conversations
# ==========================================

def retrieve(query, top_k=5):

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        results.append({
            "score": float(score),
            "customer": df.iloc[idx]["customer_message"],
            "support": df.iloc[idx]["support_response"]
        })

    return results


# ==========================================
# 4. Generate Gemini response
# ==========================================

def generate_response(query):

    # Retrieve similar historical conversations
    results = retrieve(query, 5)

    # Decide AUTO-HANDLED or ESCALATED
    decision, reason = decide_escalation(
        query,
        results
    )

    # Build context for Gemini
    context = ""

    for i, result in enumerate(results, start=1):

        context += f"""
Example {i}
Customer: {result['customer']}
Support: {result['support']}
Similarity: {result['score']:.4f}

"""

    # Gemini prompt
    prompt = f"""
You are an Apple customer support assistant.

Use the historical AppleSupport conversations below
to answer the customer's question.

Rules:
1. Give a helpful and professional response.
2. Keep the response concise.
3. Use the historical conversations as supporting context.
4. Do not invent information.
5. If the historical context is insufficient, say that
   further support may be required.
6. Do not mention FAISS, embeddings, or internal systems.
7. Do not mention that you are an AI.

Historical AppleSupport conversations:

{context}

Customer question:

{query}

Generate the best possible customer support response.
"""

    # Call Gemini
    response = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )

    answer = response.output_text

    return answer, results, decision, reason


# ==========================================
# 5. Main application
# ==========================================

while True:

    query = input(
        "\nEnter customer message "
        "(type 'exit' to stop): "
    )

    if query.lower() == "exit":

        print("Goodbye!")
        break

    # Generate complete response
    answer, results, decision, reason = generate_response(
        query
    )

    # ======================================
    # Display retrieved conversations
    # ======================================

    print("\n================================")
    print("TOP 5 SIMILAR CONVERSATIONS")
    print("================================")

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Result {i} ---")

        print(
            "Similarity:",
            round(result["score"], 4)
        )

        print(
            "Customer:",
            result["customer"]
        )

        print(
            "Support :",
            result["support"]
        )

    # ======================================
    # Display Gemini response
    # ======================================

    print("\n================================")
    print("GEMINI RESPONSE")
    print("================================")

    print(answer)

    # ======================================
    # Display decision
    # ======================================

    print("\n================================")
    print("SUPPORT DECISION")
    print("================================")

    print("Status:", decision)
    print("Reason:", reason)