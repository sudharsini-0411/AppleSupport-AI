import pandas as pd

input_file = "data/processed/golden_evaluation_context_reviewed.csv"
output_file = "data/processed/golden_evaluation_final.csv"

df = pd.read_csv(input_file)

def correct_intent(text):
    text = str(text).lower()

    # REFUND
    if any(x in text for x in [
        "refund", "money back", "give my money back",
        "want my money back", "return my money"
    ]):
        return "refund"

    # PAYMENT / BILLING
    if any(x in text for x in [
        "charged", "charge", "billing", "bill",
        "payment", "credit card", "debit card",
        "paypal", "purchase"
    ]):
        return "payment_billing"

    # SUBSCRIPTION
    if any(x in text for x in [
        "subscription", "subscribe", "unsubscribe",
        "renewal", "cancel subscription"
    ]):
        return "subscription"

    # ACCOUNT
    if any(x in text for x in [
        "apple id", "icloud", "account",
        "login", "log in", "sign in",
        "password", "forgot password",
        "account locked", "account hacked"
    ]):
        return "account_issue"

    # SOFTWARE UPDATE
    if any(x in text for x in [
        "ios update", "software update",
        "software upgrade", "ios update",
        "upgrade", "downgrade",
        "after the update", "since the update",
        "new ios", "new version"
    ]):
        return "software_update"

    # BATTERY
    if any(x in text for x in [
        "battery", "battery drain",
        "battery draining", "battery life",
        "not charging", "charging problem",
        "charging issue", "overheating"
    ]):
        return "battery_issue"

    # CONNECTIVITY
    if any(x in text for x in [
        "wifi", "wi-fi", "bluetooth",
        "internet", "network",
        "no signal", "mobile data",
        "connection problem", "not connecting",
        "disconnecting"
    ]):
        return "connectivity_issue"

    # APP
    if any(x in text for x in [
        "app", "apps", "application",
        "crash", "crashing",
        "app store", "itunes"
    ]):
        return "app_issue"

    # ORDER / DELIVERY
    if any(x in text for x in [
        "delivery", "deliver", "delivered",
        "order", "shipping", "shipment",
        "tracking number", "where is my order"
    ]):
        return "order_delivery"

    # DEVICE / HARDWARE
    if any(x in text for x in [
        "iphone", "ipad", "macbook", "mac",
        "screen", "keyboard", "camera",
        "speaker", "microphone",
        "touchid", "touch id",
        "button", "headphones",
        "earphones", "phone",
        "device", "hardware",
        "not turning on", "won't turn on"
    ]):
        return "device_issue"

    return "other"


# Keep original label
df["original_intent"] = df["actual_intent"]

# Generate corrected label
df["actual_intent"] = df["customer_message"].apply(correct_intent)

df.to_csv(output_file, index=False)

print("======================================")
print("GOLDEN SET CORRECTION COMPLETED")
print("======================================")
print("Original rows :", len(df))
print("Output file   :", output_file)

print("\nCorrected distribution:")
print(df["actual_intent"].value_counts())

print("\nLabel changes:")
print(
    (df["original_intent"] != df["actual_intent"]).sum(),
    "labels changed"
)