def decide_escalation(query, retrieved_results):

    # Get the best FAISS similarity score
    best_score = retrieved_results[0]["score"]

    # Keywords that usually need human support
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

    query_lower = query.lower()

    # Rule 1: Sensitive issue
    for keyword in escalation_keywords:
        if keyword in query_lower:
            return "ESCALATED", f"Sensitive issue detected: {keyword}"

    # Rule 2: Low similarity
    if best_score < 0.60:
        return "ESCALATED", "No sufficiently similar historical conversation found"

    # Otherwise handle automatically
    return "AUTO-HANDLED", "Relevant historical conversations found"


# Test
if __name__ == "__main__":

    query = input("Enter customer message: ")

    # Example FAISS result
    results = [
        {"score": 0.85}
    ]

    decision, reason = decide_escalation(query, results)

    print("\nDecision:", decision)
    print("Reason:", reason)