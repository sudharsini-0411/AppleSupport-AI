import pandas as pd
import re

input_file = "data/processed/golden_evaluation_set.csv"
output_file = "data/processed/golden_evaluation_set.csv"

df = pd.read_csv(input_file)


def suggest_intent(text):
    text = str(text).lower()

    if any(x in text for x in ["refund", "money back", "return my money"]):
        return "refund"

    if any(x in text for x in ["battery", "drain", "charging", "charge"]):
        return "battery_issue"

    if any(x in text for x in ["update", "ios", "software"]):
        return "software_update"

    if any(x in text for x in ["app", "application"]):
        return "app_issue"

    if any(x in text for x in ["account", "login", "password", "sign in"]):
        return "account_issue"

    if any(x in text for x in ["payment", "charged", "billing", "price"]):
        return "payment_billing"

    if any(x in text for x in ["subscription", "unsubscribe", "cancel subscription"]):
        return "subscription"

    if any(x in text for x in ["wifi", "wi-fi", "network", "internet", "signal"]):
        return "connectivity_issue"

    if any(x in text for x in ["order", "delivery", "shipping", "package"]):
        return "order_delivery"

    if any(x in text for x in ["iphone", "ipad", "mac", "device", "screen"]):
        return "device_issue"

    return "other"


df["suggested_intent"] = df["customer_message"].apply(
    suggest_intent
)

# Suggested expected response = historical support response
df["suggested_response"] = df["support_response"]

# Suggested decision
df["suggested_decision"] = "AUTO-HANDLED"

# Sensitive issues → escalation
sensitive_words = [
    "hacked",
    "fraud",
    "stolen",
    "lawsuit",
    "police",
    "chargeback"
]

for i, text in enumerate(df["customer_message"]):

    text = str(text).lower()

    if any(word in text for word in sensitive_words):
        df.loc[i, "suggested_decision"] = "ESCALATED"


df.to_csv(output_file, index=False)

print("Golden Set updated successfully!")
print("Rows:", len(df))
print("\nColumns:")
print(df.columns.tolist())