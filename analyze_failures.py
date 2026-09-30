import sys
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# -----------------------------------------------
# 1. Load evaluation results
# -----------------------------------------------

results_df = pd.read_csv("data/processed/rag_gemini_evaluation.csv")

print(f"Total evaluated examples: {len(results_df)}")
print(f"Avg overall score       : {results_df['avg_judge_score'].mean():.4f} / 5\n")

# -----------------------------------------------
# 2. Identify top 5 failures (lowest avg_judge_score)
# -----------------------------------------------

failures = results_df.sort_values("avg_judge_score", ascending=True).head(5)

print("=" * 70)
print("  TOP 5 FAILURES — RAG + GEMINI SYSTEM")
print("=" * 70)

for rank, (_, row) in enumerate(failures.iterrows(), start=1):

    print(f"\n{'─' * 70}")
    print(f"FAILURE #{rank}  |  Avg Score: {row['avg_judge_score']} / 5  |  Intent: {row['corrected_intent']}")
    print(f"{'─' * 70}")

    print(f"\nCustomer Message:\n  {row['customer_message']}")
    print(f"\nExpected Response:\n  {row['expected_response']}")
    print(f"\nGenerated Response:\n  {row['generated_response']}")

    print(f"\nJudge Scores:")
    print(f"  Relevance    : {row['relevance']} / 5")
    print(f"  Correctness  : {row['correctness']} / 5")
    print(f"  Fluency      : {row['fluency']} / 5")
    print(f"  Completeness : {row['completeness']} / 5")
    print(f"  FAISS Score  : {row['best_faiss_score']}")

    # Diagnose the failure reason
    reasons = []

    if float(row["best_faiss_score"]) < 0.60:
        reasons.append("Low FAISS similarity — retrieved conversations were not relevant enough")

    if int(row["relevance"]) <= 2:
        reasons.append("Response did not address the customer's actual problem")

    if int(row["correctness"]) <= 2:
        reasons.append("Response contained inaccurate or unhelpful information")

    if int(row["completeness"]) <= 2:
        reasons.append("Response was incomplete — key parts of the issue were not addressed")

    if int(row["fluency"]) <= 2:
        reasons.append("Response was unclear or unprofessional in tone")

    if not reasons:
        reasons.append("Marginal failure — scores were low across all dimensions without a single dominant cause")

    print(f"\nFailure Diagnosis:")
    for r in reasons:
        print(f"  • {r}")

print(f"\n{'=' * 70}")

# -----------------------------------------------
# 3. Failure pattern summary
# -----------------------------------------------

print("\n")
print("=" * 70)
print("  FAILURE PATTERN SUMMARY")
print("=" * 70)

low_faiss   = (results_df["best_faiss_score"] < 0.60).sum()
low_rel     = (results_df["relevance"] <= 2).sum()
low_correct = (results_df["correctness"] <= 2).sum()
low_fluency = (results_df["fluency"] <= 2).sum()
low_complete= (results_df["completeness"] <= 2).sum()

total = len(results_df)

print(f"Low FAISS similarity (<0.60)  : {low_faiss:>3} / {total} ({100*low_faiss/total:.1f}%)")
print(f"Low Relevance (≤2)            : {low_rel:>3} / {total} ({100*low_rel/total:.1f}%)")
print(f"Low Correctness (≤2)          : {low_correct:>3} / {total} ({100*low_correct/total:.1f}%)")
print(f"Low Fluency (≤2)              : {low_fluency:>3} / {total} ({100*low_fluency/total:.1f}%)")
print(f"Low Completeness (≤2)         : {low_complete:>3} / {total} ({100*low_complete/total:.1f}%)")

# Failures by intent
print("\nAvg score by intent:")
intent_scores = (
    results_df.groupby("corrected_intent")["avg_judge_score"]
    .agg(["mean", "count"])
    .rename(columns={"mean": "avg_score", "count": "n"})
    .sort_values("avg_score", ascending=True)
)
print(intent_scores.to_string())

# -----------------------------------------------
# 4. Save failure report
# -----------------------------------------------

failures.to_csv(
    "data/processed/top5_failures.csv",
    index=False
)

print("\nTop 5 failures saved: data/processed/top5_failures.csv")
