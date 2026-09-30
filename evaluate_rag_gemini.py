import os
import time
import re
import pandas as pd
import faiss

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY is not set. Add it to your .env file."
    )

MODEL = "gemini-3.6-flash"

print("=" * 60)
print("RAG + GEMINI EVALUATION")
print("=" * 60)
print("Gemini model:", MODEL)

client = genai.Client(api_key=api_key)


# ============================================================
# 2. LOAD RAG FILES
# ============================================================

print("\nLoading FAISS index...")

index = faiss.read_index(
    "data/processed/apple_support.faiss"
)

print("FAISS vectors:", index.ntotal)


print("\nLoading RAG dataset...")

rag_df = pd.read_csv(
    "data/processed/apple_support_rag.csv"
)

print("RAG rows:", len(rag_df))


print("\nLoading Sentence Transformer...")

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# 3. LOAD GOLDEN SET
# ============================================================

golden_df = pd.read_csv(
    "data/processed/golden_evaluation_final.csv"
)

print("\nGolden Set loaded:", len(golden_df))


required_columns = [
    "tweet_id",
    "customer_message",
    "support_response",
    "corrected_intent"
]

missing_columns = [
    col
    for col in required_columns
    if col not in golden_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in Golden Set: {missing_columns}"
    )

print("RAG system ready.")


# ============================================================
# 4. OUTPUT FILE + RESUME PREVIOUS RESULTS
# ============================================================

output_file = (
    "data/processed/"
    "rag_gemini_evaluation.csv"
)

TOTAL_EVALUATION = 50


if os.path.exists(output_file):

    previous_results = pd.read_csv(
        output_file
    )

    completed_ids = set(
        previous_results["tweet_id"]
        .astype(str)
    )

    print(
        "\nPrevious successful evaluations:",
        len(previous_results)
    )

else:

    previous_results = pd.DataFrame()

    completed_ids = set()

    print(
        "\nNo previous evaluation results found."
    )


# Select only examples that are not already completed
eval_df = golden_df[
    ~golden_df["tweet_id"]
    .astype(str)
    .isin(completed_ids)
].head(TOTAL_EVALUATION).copy()


print("\n" + "=" * 60)
print(
    f"New examples to evaluate: {len(eval_df)}"
)
print(
    f"Target total evaluations: {TOTAL_EVALUATION}"
)
print("=" * 60)


# ============================================================
# 5. RETRIEVE SIMILAR CONVERSATIONS
# ============================================================

def retrieve(query, top_k=5):

    embedding = embedder.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(
        embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        results.append({
            "score": float(score),

            "customer": str(
                rag_df.iloc[idx]["customer_message"]
            ),

            "support": str(
                rag_df.iloc[idx]["support_response"]
            )
        })

    return results


# ============================================================
# 6. GENERATE RAG RESPONSE
# ============================================================

def generate_response(query):

    retrieved_results = retrieve(
        query,
        top_k=5
    )

    context = ""

    for i, result in enumerate(
        retrieved_results,
        start=1
    ):

        context += f"""
Example {i}

Customer:
{result["customer"]}

Historical Support Response:
{result["support"]}

Similarity:
{result["score"]:.4f}

"""


    prompt = f"""
You are an Apple customer support assistant.

Use the historical AppleSupport conversations below
to help answer the customer's question.

Rules:

1. Give a helpful and professional response.
2. Keep the response concise, around 2-4 sentences.
3. Use the historical conversations as supporting context.
4. Do not invent information.
5. If the historical examples do not provide enough information,
   clearly state that further support may be required.
6. Do not mention FAISS.
7. Do not mention embeddings.
8. Do not mention internal systems.
9. Do not mention that you are an AI.

Historical AppleSupport conversations:

{context}

Customer question:

{query}

Generate the best possible customer support response.
"""


    response = client.interactions.create(
        model=MODEL,
        input=prompt
    )

    generated_text = response.output_text

    if not generated_text:

        raise ValueError(
            "Gemini returned an empty response."
        )

    return (
        generated_text.strip(),
        retrieved_results
    )


# ============================================================
# 7. LLM-AS-A-JUDGE
# ============================================================

def judge_response(
    customer_message,
    generated_response,
    expected_response
):

    judge_prompt = f"""
You are an expert evaluator for customer support quality.

Evaluate the Generated Response using the four criteria below.

Give an integer score from 1 to 5 for each criterion.

1 = Very poor
2 = Poor
3 = Acceptable
4 = Good
5 = Excellent


Customer Message:

{customer_message}


Expected Response:

{expected_response}


Generated Response:

{generated_response}


Criteria:

Relevance:
Does the generated response directly address the customer's problem?

Correctness:
Is the information accurate and helpful?

Fluency:
Is the response clear, professional, and well-written?

Completeness:
Does the response sufficiently address the customer's issue?


IMPORTANT:

Reply ONLY using exactly this format:

Relevance: <number>
Correctness: <number>
Fluency: <number>
Completeness: <number>
"""


    response = client.interactions.create(
        model=MODEL,
        input=judge_prompt
    )

    text = response.output_text.strip()

    scores = {}

    patterns = {

        "Relevance":
            r"Relevance\s*:\s*([1-5])",

        "Correctness":
            r"Correctness\s*:\s*([1-5])",

        "Fluency":
            r"Fluency\s*:\s*([1-5])",

        "Completeness":
            r"Completeness\s*:\s*([1-5])"
    }


    for criterion, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            scores[criterion] = int(
                match.group(1)
            )


    required_scores = [
        "Relevance",
        "Correctness",
        "Fluency",
        "Completeness"
    ]


    for criterion in required_scores:

        if criterion not in scores:

            scores[criterion] = 0


    return scores


# ============================================================
# 8. EVALUATION LOOP
# ============================================================

records = []

quota_reached = False


for position, (index_number, row) in enumerate(
    eval_df.iterrows(),
    start=1
):

    customer_msg = str(
        row["customer_message"]
    )

    expected_resp = str(
        row["support_response"]
    )

    corrected_intent = str(
        row["corrected_intent"]
    )


    print(
        f"\nProcessing example "
        f"{position}/{len(eval_df)}"
    )

    print(
        "Customer:",
        customer_msg[:100]
    )


    try:

        # ----------------------------------------------------
        # Generate RAG response
        # ----------------------------------------------------

        generated_response, retrieved = generate_response(
            customer_msg
        )


        best_score = (
            retrieved[0]["score"]
            if retrieved
            else 0.0
        )


        # ----------------------------------------------------
        # Judge response
        # ----------------------------------------------------

        scores = judge_response(
            customer_msg,
            generated_response,
            expected_resp
        )


        # ----------------------------------------------------
        # Calculate average score
        # ----------------------------------------------------

        valid_scores = [

            scores["Relevance"],

            scores["Correctness"],

            scores["Fluency"],

            scores["Completeness"]
        ]


        valid_scores = [

            score
            for score in valid_scores
            if score > 0
        ]


        if valid_scores:

            avg_score = round(
                sum(valid_scores) /
                len(valid_scores),
                2
            )

        else:

            avg_score = 0.0


        # ----------------------------------------------------
        # Create successful record
        # ----------------------------------------------------

        record = {

            "tweet_id":
                row["tweet_id"],

            "customer_message":
                customer_msg,

            "corrected_intent":
                corrected_intent,

            "expected_response":
                expected_resp,

            "generated_response":
                generated_response,

            "best_faiss_score":
                round(
                    best_score,
                    4
                ),

            "relevance":
                scores["Relevance"],

            "correctness":
                scores["Correctness"],

            "fluency":
                scores["Fluency"],

            "completeness":
                scores["Completeness"],

            "avg_judge_score":
                avg_score
        }


        records.append(record)


        # ----------------------------------------------------
        # SAVE IMMEDIATELY
        # ----------------------------------------------------

        current_results = pd.DataFrame(
            records
        )


        if not previous_results.empty:

            current_results = pd.concat(
                [
                    previous_results,
                    current_results
                ],
                ignore_index=True
            )


        current_results = current_results.drop_duplicates(
            subset=["tweet_id"],
            keep="last"
        )


        current_results.to_csv(
            output_file,
            index=False
        )


        print(
            "\nGenerated response:"
        )

        print(
            generated_response
        )


        print(
            "\nJudge scores:"
        )

        print(
            "Relevance   :",
            scores["Relevance"]
        )

        print(
            "Correctness :",
            scores["Correctness"]
        )

        print(
            "Fluency     :",
            scores["Fluency"]
        )

        print(
            "Completeness:",
            scores["Completeness"]
        )

        print(
            "Average     :",
            avg_score
        )

        print(
            "FAISS score :",
            round(
                best_score,
                4
            )
        )

        print(
            "Progress saved."
        )


    except Exception as e:

        print(
            f"\n[ERROR] Example {position}:"
        )

        print(
            type(e).__name__,
            str(e)
        )


        # ----------------------------------------------------
        # STOP WHEN GEMINI QUOTA IS REACHED
        # ----------------------------------------------------

        error_text = str(e).lower()

        if (
            "429" in error_text
            or
            "quota" in error_text
            or
            "rate limit" in error_text
        ):

            print(
                "\nGemini quota reached."
            )

            print(
                "Stopping evaluation."
            )

            print(
                "Successful results have already been saved."
            )

            quota_reached = True

            break


    # --------------------------------------------------------
    # Small delay
    # --------------------------------------------------------

    time.sleep(2)


# ============================================================
# 9. LOAD FINAL SAVED RESULTS
# ============================================================

if os.path.exists(output_file):

    results_df = pd.read_csv(
        output_file
    )

else:

    results_df = pd.DataFrame()


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

print("\n\n")

print("=" * 60)

print(
    "RAG + GEMINI — LLM-as-a-Judge Evaluation"
)

print(
    f"Total successful evaluations: "
    f"{len(results_df)}"
)

print("=" * 60)


if len(results_df) == 0:

    print(
        "\nNo successful evaluations were completed."
    )

else:

    print(
        f"Avg Relevance    : "
        f"{results_df['relevance'].mean():.4f} / 5"
    )

    print(
        f"Avg Correctness  : "
        f"{results_df['correctness'].mean():.4f} / 5"
    )

    print(
        f"Avg Fluency      : "
        f"{results_df['fluency'].mean():.4f} / 5"
    )

    print(
        f"Avg Completeness : "
        f"{results_df['completeness'].mean():.4f} / 5"
    )

    print(
        f"Avg Overall Score: "
        f"{results_df['avg_judge_score'].mean():.4f} / 5"
    )

    print(
        f"Avg FAISS Score  : "
        f"{results_df['best_faiss_score'].mean():.4f}"
    )


    # ========================================================
    # SCORE DISTRIBUTION
    # ========================================================

    print(
        "\nScore distribution:"
    )


    def score_band(score):

        if score <= 2:

            return "Poor (1-2)"

        elif score <= 3:

            return "Fair (3)"

        elif score <= 4:

            return "Good (4)"

        else:

            return "Excellent (5)"


    results_df["score_band"] = (
        results_df["avg_judge_score"]
        .apply(score_band)
    )


    print(
        results_df[
            "score_band"
        ]
        .value_counts()
        .reindex(
            [
                "Poor (1-2)",
                "Fair (3)",
                "Good (4)",
                "Excellent (5)"
            ],
            fill_value=0
        )
        .to_string()
    )


# ============================================================
# 11. SAVE FINAL RESULTS
# ============================================================

results_df.to_csv(
    output_file,
    index=False
)


print("\n")

print(
    "Results saved to:"
)

print(
    output_file
)

print("=" * 60)