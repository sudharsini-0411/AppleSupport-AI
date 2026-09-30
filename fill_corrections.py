import pandas as pd

input_file = "data/processed/golden_review_working.csv"
output_file = "data/processed/golden_review_working.csv"

# Load the working Golden Set
df = pd.read_csv(input_file)

# Make sure corrected_intent is a text column
df["corrected_intent"] = df["corrected_intent"].fillna("").astype(str)

# Corrections for rows 1-40
corrections = {
    # Rows 1-17
    1: "other",
    2: "refund",
    3: "app_issue",
    4: "connectivity_issue",
    5: "device_issue",
    6: "other",
    7: "other",
    8: "software_update",
    9: "device_issue",
    10: "device_issue",
    11: "software_update",
    12: "app_issue",
    13: "other",
    14: "software_update",
    15: "connectivity_issue",
    16: "device_issue",
    17: "other",

    # Rows 18-40
    18: "software_update",
    19: "battery_issue",
    20: "app_issue",
    21: "software_update",
    22: "device_issue",
    23: "other",
    24: "other",
    25: "other",
    26: "other",
    27: "device_issue",
    28: "connectivity_issue",
    29: "battery_issue",
    30: "device_issue",
    31: "account_issue",
    32: "software_update",
    33: "device_issue",
    34: "account_issue",
    35: "device_issue",
    36: "connectivity_issue",
    37: "software_update",
    38: "device_issue",
    39: "connectivity_issue",
    40: "software_update"
}

# Fill ONLY corrected_intent
for row_number, intent in corrections.items():
    df.iloc[
        row_number - 1,
        df.columns.get_loc("corrected_intent")
    ] = intent

# Save
df.to_csv(output_file, index=False)

print("======================================")
print("CORRECTIONS ADDED")
print("======================================")
print("Total rows:", len(df))
print("Rows corrected:", len(corrections))
print()

# Show rows 1-40
print(
    df.iloc[:40][
        ["customer_message", "actual_intent", "corrected_intent"]
    ].to_string(index=False)
)

print()
print("Original actual_intent was NOT changed.")
print("Saved:", output_file)