import pandas as pd
import re

input_file = "data/processed/apple_support_clean.csv"
output_file = "data/processed/apple_support_labeled_v2.csv"

df = pd.read_csv(input_file)

print("Loaded conversations:", len(df))


def detect_intent(text):
    text = str(text).lower()

    # --------------------------------------------------
    # 1. REFUND
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "refund",
        "money back",
        "give my money back",
        "want my money back",
        "get my money back",
        "return my money",
        "charged me and i want",
        "charged me but"
    ]):
        return "refund"

    # --------------------------------------------------
    # 2. PAYMENT / BILLING
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "payment",
        "billing",
        "bill",
        "charged",
        "charge me",
        "credit card",
        "debit card",
        "paypal",
        "purchase",
        "payment failed",
        "charged twice",
        "wrong charge",
        "unknown charge",
        "unexpected charge"
    ]):
        return "payment_billing"

    # --------------------------------------------------
    # 3. SUBSCRIPTION
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "subscription",
        "subscribe",
        "unsubscribe",
        "cancel subscription",
        "subscription cancelled",
        "renewal",
        "renew my subscription"
    ]):
        return "subscription"

    # --------------------------------------------------
    # 4. ACCOUNT
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "apple id",
        "icloud account",
        "my account",
        "account locked",
        "account disabled",
        "can't login",
        "cannot login",
        "can't log in",
        "cannot log in",
        "login problem",
        "sign in",
        "signin",
        "password",
        "forgot password",
        "username",
        "hacked account",
        "account hacked"
    ]):
        return "account_issue"

    # --------------------------------------------------
    # 5. SOFTWARE UPDATE
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "ios update",
        "ios",
        "software update",
        "software upgrade",
        "update",
        "updated",
        "upgrade",
        "downgrade",
        "latest version",
        "new version",
        "after the update",
        "since the update"
    ]):
        return "software_update"

    # --------------------------------------------------
    # 6. BATTERY
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "battery",
        "battery life",
        "battery drain",
        "battery draining",
        "draining battery",
        "battery dies",
        "battery dying",
        "won't hold charge",
        "not holding charge",
        "charging problem",
        "charging issue",
        "charging slowly",
        "not charging",
        "overheating"
    ]):
        return "battery_issue"

    # --------------------------------------------------
    # 7. CONNECTIVITY
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "wifi",
        "wi-fi",
        "internet",
        "network",
        "no signal",
        "signal problem",
        "mobile data",
        "bluetooth",
        "connection problem",
        "connection issue",
        "can't connect",
        "cannot connect",
        "not connecting",
        "disconnecting"
    ]):
        return "connectivity_issue"

    # --------------------------------------------------
    # 8. APP ISSUE
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "app",
        "apps",
        "application",
        "crash",
        "crashes",
        "crashing",
        "app store",
        "itunes",
        "whatsapp",
        "instagram",
        "facebook",
        "snapchat",
        "twitter",
        "reminders",
        "music app",
        "mail app",
        "calendar app"
    ]):
        return "app_issue"

    # --------------------------------------------------
    # 9. ORDER / DELIVERY
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "delivery",
        "deliver",
        "delivered",
        "order",
        "shipping",
        "shipment",
        "arrive",
        "arrival",
        "tracking",
        "tracking number",
        "where is my order",
        "when will it arrive"
    ]):
        return "order_delivery"

    # --------------------------------------------------
    # 10. DEVICE / HARDWARE
    # --------------------------------------------------
    if any(phrase in text for phrase in [
        "iphone screen",
        "ipad screen",
        "macbook screen",
        "broken screen",
        "screen replacement",
        "screen cracked",
        "keyboard",
        "camera",
        "speaker",
        "microphone",
        "silent button",
        "home button",
        "power button",
        "volume button",
        "phone restart",
        "phone restarting",
        "device restart",
        "device not turning on",
        "won't turn on",
        "not turning on",
        "black screen",
        "hardware problem",
        "hardware issue"
    ]):
        return "device_issue"

    # --------------------------------------------------
    # 11. GENERAL
    # --------------------------------------------------
    return "other"


# Apply classification
df["intent"] = df["customer_message"].apply(detect_intent)


# Save
df.to_csv(output_file, index=False)


print("\n======================================")
print("IMPROVED LABELING COMPLETED")
print("======================================")

print("\nIntent distribution:")
print(df["intent"].value_counts())

print("\nSaved successfully!")
print("File:", output_file)