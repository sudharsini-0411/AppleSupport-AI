import os
import time
import pandas as pd
from google import genai

input_file = "data/processed/golden_evaluation_context_reviewed.csv"
output_file = "data/processed/golden_llm_suggestions.csv"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

df = pd.read_csv(input_file)

# Resume from previous progress if file exists
if os.path.exists(output_file):
    results = pd.read_csv(output_file)
    processed_ids = set(results["tweet_id"].astype(str))
    print("Existing results:", len(results))
else:
    results = pd.DataFrame(columns=[
        "tweet_id",
        "customer_message",
        "support_response",
        "actual_intent",
        "llm_suggested_intent",
        "llm_reason"
    ])
    processed_ids = set()

intents = """
account_issue
app_issue
battery_issue
connectivity_issue
device_issue
order_delivery
other
payment_billing
refund
software_update
subscription
"""

for i, row in df.iterrows():

    tweet_id = str(row["tweet_id"])

    if tweet_id in processed_ids:
        continue

    print(f"\nProcessing {i + 1}/{len(df)}")

    prompt = f"""
You are reviewing a customer-support intent classification dataset.

Choose exactly ONE intent from this list:

{intents}

Customer message:
{row["customer_message"]}

Support response:
{row["support_response"]}

Conversation context:
{row.get("conversation_context", "")}

Return exactly this format:

INTENT: <one intent>
REASON: <short reason>

Do not create a new intent.
"""

    try:
        response = client.models.generate_content(
            model="gemini-3.6-flash",
            contents=prompt
        )

        text = response.text.strip()

        suggested_intent = "other"
        reason = text

        for intent in intents.strip().split("\n"):
            if f"INTENT: {intent}" in text:
                suggested_intent = intent
                break

        if "REASON:" in text:
            reason = text.split("REASON:", 1)[1].strip()

        new_row = {
            "tweet_id": tweet_id,
            "customer_message": row["customer_message"],
            "support_response": row["support_response"],
            "actual_intent": row["actual_intent"],
            "llm_suggested_intent": suggested_intent,
            "llm_reason": reason
        }

        results = pd.concat(
            [results, pd.DataFrame([new_row])],
            ignore_index=True
        )

        # SAVE AFTER EVERY ROW
        results.to_csv(output_file, index=False)

        print("LLM intent:", suggested_intent)
        print("Saved:", len(results))

        time.sleep(2)

    except Exception as e:

        error = str(e)

        print("\nERROR:")
        print(error)

        if "429" in error or "RESOURCE_EXHAUSTED" in error or "quota" in error.lower():
            print("\nGemini quota reached.")
            print("Progress has been saved.")
            print("Run this script again after the quota resets.")
            break

        print("Skipping this row...")
        continue

print("\n======================================")
print("PROCESS COMPLETED / CHECKPOINT SAVED")
print("======================================")
print("Rows saved:", len(results))
print("File:", output_file)