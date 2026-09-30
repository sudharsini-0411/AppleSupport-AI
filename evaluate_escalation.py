import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


# ============================================================
# 1. LOAD FILES
# ============================================================

golden_file = "data/processed/golden_evaluation_final.csv"
rag_file = "data/processed/apple_support_rag.csv"
faiss_file = "data/processed/apple_support.faiss"

golden_df = pd.read_csv(golden_file)
rag_df = pd.read_csv(rag_file)

index = faiss.read_index(faiss_file)

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("=" * 60)
print("ESCALATION EVALUATION - SELF MATCH REMOVED")
print("=" * 60)

print("Golden Set:", len(golden_df))
print("FAISS vectors:", index.ntotal)


# ============================================================
# 2. ESCALATION RULES
# ============================================================

escalation_keywords = [
    "legal",
    "lawsuit",
    "chargeback",
    "fraud",
    "hacked",
    "stolen",
    "police",
    "security",
    "personal information"
]


def decide_escalation(query, retrieved_results):

    best_score = (
        retrieved_results[0]["score"]
        if retrieved_results
        else 0.0
    )

    query_lower = query.lower()

    # Sensitive issue
    for keyword in escalation_keywords:

        if keyword in query_lower:

            return (
                "ESCALATED",
                f"Sensitive issue detected: {keyword}",
                best_score
            )

    # Low similarity
    if best_score < 0.60:

        return (
            "ESCALATED",
            "No sufficiently similar historical conversation found",
            best_score
        )

    # Normal issue
    return (
        "AUTO-HANDLED",
        "Relevant historical conversations found",
        best_score
    )


# ============================================================
# 3. RETRIEVE WITHOUT SELF-MATCH
# ============================================================

def retrieve(query, golden_tweet_id, top_k=5):

    embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    # Search more results because the first result
    # may be the exact Golden Set conversation.
    search_k = top_k + 20

    scores, indices = index.search(
        embedding,
        search_k
    )

    results = []

    golden_id = str(golden_tweet_id)

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        rag_id = str(
            rag_df.iloc[idx]["tweet_id"]
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # Remove the Golden Set example itself
        # ----------------------------------------------------

        if rag_id == golden_id:
            continue

        results.append({

            "score": float(score),

            "customer": str(
                rag_df.iloc[idx]["customer_message"]
            ),

            "support": str(
                rag_df.iloc[idx]["support_response"]
            ),

            "tweet_id": rag_id
        })

        if len(results) == top_k:
            break

    return results


# ============================================================
# 4. EVALUATE GOLDEN SET
# ============================================================

results = []


for i, row in golden_df.iterrows():

    customer_message = str(
        row["customer_message"]
    )

    retrieved = retrieve(
        customer_message,
        row["tweet_id"],
        top_k=5
    )

    decision, reason, score = decide_escalation(
        customer_message,
        retrieved
    )

    results.append({

        "tweet_id":
            row["tweet_id"],

        "customer_message":
            customer_message,

        "corrected_intent":
            row["corrected_intent"],

        "decision":
            decision,

        "reason":
            reason,

        "best_faiss_score":
            round(score, 4)
    })


# ============================================================
# 5. SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

output_file = (
    "data/processed/"
    "escalation_evaluation_no_self_match.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 6. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("ESCALATION SUMMARY")
print("=" * 60)

print("\nDecision distribution:")

print(
    results_df["decision"]
    .value_counts()
)


print("\nReason distribution:")

print(
    results_df["reason"]
    .value_counts()
)


print(
    "\nFAISS score statistics:"
)

print(
    results_df["best_faiss_score"]
    .describe()
)


print(
    "\nAverage FAISS score:",
    round(
        results_df["best_faiss_score"].mean(),
        4
    )
)


print(
    "\nResults saved to:"
)

print(output_file)

print("=" * 60)