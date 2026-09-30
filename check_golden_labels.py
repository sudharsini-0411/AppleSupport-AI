import pandas as pd

input_file = "data/processed/golden_evaluation_context_reviewed.csv"
output_file = "data/processed/golden_label_check.csv"

df = pd.read_csv(input_file)

def suggest_intent(text):
    text = str(text).lower()

    if any(x in text for x in ["refund", "money back", "give my money back"]):
        return "refund"

    if any(x in text for x in ["payment", "billing", "charged", "charge", "credit card", "debit card"]):
        return "payment_billing"

    if any(x in text for x in ["subscription", "subscribe", "unsubscribe", "renewal"]):
        return "subscription"

    if any(x in text for x in ["apple id", "icloud", "account", "password", "login", "log in", "sign in", "hacked"]):
        return "account_issue"

    if any(x in text for x in ["ios update", "software update", "updated", "upgrading", "upgrade", "ios 11", "macos"]):
        return "software_update"

    if any(x in text for x in ["battery", "charging", "not charging", "battery drain", "overheating"]):
        return "battery_issue"

    if any(x in text for x in ["wifi", "wi-fi", "bluetooth", "4g", "network", "signal", "mobile data", "connection"]):
        return "connectivity_issue"

    if any(x in text for x in ["app", "apps", "twitter app", "reminders", "music app", "mail app"]):
        return "app_issue"

    if any(x in text for x in ["delivery", "shipping", "shipment", "tracking", "order", "arrive"]):
        return "order_delivery"

    if any(x in text for x in ["screen", "keyboard", "camera", "speaker", "button", "touchid", "headphones", "macbook", "iphone", "ipad"]):
        return "device_issue"

    return "other"


df["suggested_intent"] = df["customer_message"].apply(suggest_intent)

df["match"] = (
    df["actual_intent"] == df["suggested_intent"]
)

df.to_csv(output_file, index=False)

print("======================================")
print("GOLDEN LABEL CHECK COMPLETED")
print("======================================")
print("Total rows:", len(df))
print("Matching:", df["match"].sum())
print("Potential mismatches:", (~df["match"]).sum())
print("Saved:", output_file)