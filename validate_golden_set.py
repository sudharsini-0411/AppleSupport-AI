import pandas as pd

FILE = "data/processed/golden_evaluation_set.csv"

df = pd.read_csv(FILE)

print("========================================")
print("GOLDEN SET VALIDATION")
print("========================================")

# 1. Check number of rows
print("\nTotal rows:", len(df))

if len(df) == 200:
    print("✅ Correct: 200 rows")
else:
    print("❌ Expected 200 rows")

# 2. Check required columns
required_columns = [
    "tweet_id",
    "customer_message",
    "support_response",
    "actual_intent",
    "expected_response",
    "expected_decision"
]

print("\nChecking columns...")

for column in required_columns:
    if column in df.columns:
        print(f"✅ {column}")
    else:
        print(f"❌ Missing: {column}")

# 3. Check missing values
print("\n========================================")
print("MISSING VALUES")
print("========================================")

for column in required_columns:
    missing = df[column].isna().sum()

    if missing == 0:
        print(f"✅ {column}: No missing values")
    else:
        print(f"❌ {column}: {missing} missing")

# 4. Check valid intents
valid_intents = [
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

invalid_intents = df[
    ~df["actual_intent"].isin(valid_intents)
]

print("\n========================================")
print("INTENT CHECK")
print("========================================")

if len(invalid_intents) == 0:
    print("✅ All intents are valid")
else:
    print("❌ Invalid intents:", len(invalid_intents))
    print(invalid_intents["actual_intent"].value_counts())

# 5. Check decisions
valid_decisions = [
    "AUTO-HANDLED",
    "ESCALATED"
]

invalid_decisions = df[
    ~df["expected_decision"].isin(valid_decisions)
]

print("\n========================================")
print("DECISION CHECK")
print("========================================")

if len(invalid_decisions) == 0:
    print("✅ All decisions are valid")
else:
    print("❌ Invalid decisions:", len(invalid_decisions))
    print(invalid_decisions["expected_decision"].value_counts())

# 6. Intent distribution
print("\n========================================")
print("INTENT DISTRIBUTION")
print("========================================")

print(df["actual_intent"].value_counts())

# 7. Decision distribution
print("\n========================================")
print("DECISION DISTRIBUTION")
print("========================================")

print(df["expected_decision"].value_counts())

print("\n========================================")
print("VALIDATION COMPLETED")
print("========================================")