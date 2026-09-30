import os
import pandas as pd
from google import genai

input_file = "data/processed/golden_evaluation_context_reviewed.csv"
output_file = "data/processed/golden_llm_suggestions.csv"

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY is not set.")

client = genai.Client(api_key=api_key)

df = pd.read_csv(input_file)

intents = [
    "battery_issue",
    "software_update",
    "app_issue",
    "account_issue",
    "payment_billing",
    "refund",
    "subscription",
    "connectivity_issue",
    "device_issue",
    "order_delivery",
    "other"
]

def classify(row):

    prompt = f"""
You are labeling an Apple customer-support dataset.

Choose EXACTLY ONE intent from this list:

{", ".join(intents)}

Intent definitions:

battery_issue = battery drain, charging, battery life, overheating
software_update = iOS/macOS update, upgrade, downgrade, version/update problems
app_issue = application/app-specific problems
account_issue = Apple ID, iCloud, login, password, account access
payment_billing = charges, payments, billing, card/payment problems
refund = customer wants money returned or a refund
subscription = subscription, renewal, unsubscribe
connectivity_issue = WiFi, Bluetooth, mobile data, network, signal, connection
device_issue = hardware/device problems such as screen, buttons, keyboard, camera, speaker
order_delivery = order, shipping, delivery, tracking
other = none of the above or insufficient information

CONVERSATION:
{row['conversation_context']}

CURRENT CUSTOMER MESSAGE:
{row['customer_message']}

SUPPORT RESPONSE:
{row['support_response']}

Return ONLY this format:

INTENT: <one intent>
REASON: <short reason>
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    text = response.text.strip()

    intent = "other"
    reason = text

    for label in intents:
        if f"INTENT: {label}" in text:
            intent = label
            break

    if "REASON:" in text:
        reason = text.split("REASON:", 1)[1].strip()

    return intent, reason


suggestions = []
reasons = []

for i, row in df.iterrows():

    print(f"Processing {i + 1}/{len(df)}")

    intent, reason = classify(row)

    suggestions.append(intent)
    reasons.append(reason)


df["llm_suggested_intent"] = suggestions
df["llm_reason"] = reasons

df.to_csv(output_file, index=False)

print("\n======================================")
print("LLM GOLDEN LABEL SUGGESTIONS CREATED")
print("======================================")
print("Rows:", len(df))
print("Saved:", output_file)